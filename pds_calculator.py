from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import csv
import io
from pybaseball import statcast


SEASON = 2026
START_DATE = f"{SEASON}-03-25"
END_DATE = date.today().isoformat()

# Change this whenever you decide on your official qualifying rule.
MIN_BATTERS_FACED = 150

# Relievers rarely reach 150 batters faced in a season, so they get their
# own, lower qualifying bar. Adjust this as you see fit.
MIN_RELIEVER_BATTERS_FACED = 60

# Threshold for the "starters with 80+ IP" highlight list.
MIN_STARTER_IP = 80

OUTPUT_FILE = Path("pds_leaderboard.csv")


SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "foul_bunt",
    "missed_bunt",
    "hit_into_play",
}

WHIFF_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
}

CSW_DESCRIPTIONS = {
    "called_strike",
    "swinging_strike",
    "swinging_strike_blocked",
}

FIRST_PITCH_STRIKE_DESCRIPTIONS = {
    "called_strike",
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "foul_bunt",
    "missed_bunt",
    "hit_into_play",
}

WALK_EVENTS = {
    "walk",
}

STRIKEOUT_EVENTS = {
    "strikeout",
    "strikeout_double_play",
}

# Outs recorded on each event type. Used to reconstruct innings pitched
# directly from pitch-level data, without needing an external stats site.
OUT_EVENT_COUNTS = {
    "strikeout": 1,
    "strikeout_double_play": 2,
    "field_out": 1,
    "force_out": 1,
    "grounded_into_double_play": 2,
    "double_play": 2,
    "triple_play": 3,
    "fielders_choice_out": 1,
    "sac_fly": 1,
    "sac_fly_double_play": 2,
    "sac_bunt": 1,
    "sac_bunt_double_play": 2,
    "other_out": 1,
    "caught_stealing_2b": 1,
    "caught_stealing_3b": 1,
    "caught_stealing_home": 1,
    "pickoff_caught_stealing_2b": 1,
    "pickoff_caught_stealing_3b": 1,
    "pickoff_caught_stealing_home": 1,
    "pickoff_1b": 1,
    "pickoff_2b": 1,
    "pickoff_3b": 1,
}


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide while returning NaN when the denominator is zero."""
    return numerator.div(denominator.replace(0, np.nan))


def download_statcast_data() -> pd.DataFrame:
    print(f"Downloading Statcast data from {START_DATE} through {END_DATE}...")

    data = statcast(start_dt=START_DATE, end_dt=END_DATE)

    if data.empty:
        raise RuntimeError("No Statcast data was returned.")

    return data.copy()


def prepare_pitch_data(data: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "pitcher",
        "player_name",
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "description",
        "events",
        "zone",
        "bb_type",
        "launch_speed_angle",
        "estimated_woba_using_speedangle",
        "woba_value",
        "inning",
        "inning_topbot",
        "home_team",
        "away_team",
    }

    missing = required_columns.difference(data.columns)

    if missing:
        raise KeyError(
            "The Statcast download is missing these columns: "
            + ", ".join(sorted(missing))
        )

    data["pitcher"] = pd.to_numeric(data["pitcher"], errors="coerce")
    data["pitch_number"] = pd.to_numeric(
        data["pitch_number"], errors="coerce"
    )
    data["zone"] = pd.to_numeric(data["zone"], errors="coerce")
    data["inning"] = pd.to_numeric(data["inning"], errors="coerce")
    data["launch_speed_angle"] = pd.to_numeric(
        data["launch_speed_angle"], errors="coerce"
    )
    data["estimated_woba_using_speedangle"] = pd.to_numeric(
        data["estimated_woba_using_speedangle"], errors="coerce"
    )
    data["woba_value"] = pd.to_numeric(
        data["woba_value"], errors="coerce"
    )

    # Each plate appearance receives a stable identifier.
    data["pa_id"] = (
        data["game_pk"].astype("string")
        + "_"
        + data["at_bat_number"].astype("string")
    )

    data["is_swing"] = data["description"].isin(SWING_DESCRIPTIONS)
    data["is_whiff"] = data["description"].isin(WHIFF_DESCRIPTIONS)
    data["is_csw"] = data["description"].isin(CSW_DESCRIPTIONS)

    data["is_first_pitch"] = data["pitch_number"].eq(1)
    data["is_first_pitch_strike"] = (
        data["is_first_pitch"]
        & data["description"].isin(FIRST_PITCH_STRIKE_DESCRIPTIONS)
    )

    # Statcast zones 1-9 are in-zone. Other numbered zones are out-of-zone.
    data["is_outside_zone"] = data["zone"].notna() & ~data["zone"].between(1, 9)
    data["is_chase"] = data["is_outside_zone"] & data["is_swing"]

    data["is_batted_ball"] = data["bb_type"].notna()
    data["is_ground_ball"] = data["bb_type"].eq("ground_ball")

    # Statcast launch_speed_angle category 6 represents barrels.
    data["is_barrel"] = data["launch_speed_angle"].eq(6)

    # Outs recorded on this pitch's event, used to build innings pitched.
    data["outs_recorded"] = data["events"].map(OUT_EVENT_COUNTS).fillna(0)

    # Top of the inning: home team is pitching. Bottom: away team is pitching.
    data["pitcher_team"] = np.where(
        data["inning_topbot"].eq("Top"), data["home_team"], data["away_team"]
    )

    return data


def calculate_pitcher_metrics(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby(["pitcher", "player_name"], dropna=False)

    totals = grouped.agg(
        pitches=("description", "size"),
        whiffs=("is_whiff", "sum"),
        csw=("is_csw", "sum"),
        first_pitches=("is_first_pitch", "sum"),
        first_pitch_strikes=("is_first_pitch_strike", "sum"),
        outside_pitches=("is_outside_zone", "sum"),
        chases=("is_chase", "sum"),
        batted_balls=("is_batted_ball", "sum"),
        barrels=("is_barrel", "sum"),
        ground_balls=("is_ground_ball", "sum"),
    ).reset_index()

    # Use the final pitch of each plate appearance for PA-level outcomes.
    terminal = (
        data.sort_values(
            ["game_pk", "at_bat_number", "pitch_number"]
        )
        .groupby(["pitcher", "player_name", "pa_id"], as_index=False)
        .tail(1)
        .copy()
    )

    terminal["is_strikeout"] = terminal["events"].isin(STRIKEOUT_EVENTS)
    terminal["is_walk"] = terminal["events"].isin(WALK_EVENTS)

    # Approximate expected wOBA:
    # use Statcast's expected contact value when available;
    # otherwise use the PA's wOBA value for non-contact outcomes.
    terminal["expected_woba_value"] = (
        terminal["estimated_woba_using_speedangle"]
        .fillna(terminal["woba_value"])
    )

    pa_totals = (
        terminal.groupby(["pitcher", "player_name"], dropna=False)
        .agg(
            batters_faced=("pa_id", "nunique"),
            strikeouts=("is_strikeout", "sum"),
            walks=("is_walk", "sum"),
            xwoba_against=("expected_woba_value", "mean"),
        )
        .reset_index()
    )

    metrics = totals.merge(
        pa_totals,
        on=["pitcher", "player_name"],
        how="inner",
    )

    metrics["K%"] = safe_divide(
        metrics["strikeouts"], metrics["batters_faced"]
    ) * 100

    metrics["SwStr%"] = safe_divide(
        metrics["whiffs"], metrics["pitches"]
    ) * 100

    metrics["CSW%"] = safe_divide(
        metrics["csw"], metrics["pitches"]
    ) * 100

    metrics["BB%"] = safe_divide(
        metrics["walks"], metrics["batters_faced"]
    ) * 100

    metrics["F-Strike%"] = safe_divide(
        metrics["first_pitch_strikes"], metrics["first_pitches"]
    ) * 100

    metrics["Chase%"] = safe_divide(
        metrics["chases"], metrics["outside_pitches"]
    ) * 100

    metrics["Barrel%"] = safe_divide(
        metrics["barrels"], metrics["batted_balls"]
    ) * 100

    metrics["GB%"] = safe_divide(
        metrics["ground_balls"], metrics["batted_balls"]
    ) * 100

    metrics = metrics.rename(columns={"xwoba_against": "xwOBA"})

    return metrics


def calculate_teams(data: pd.DataFrame) -> pd.DataFrame:
    """Assign each pitcher their most frequently pitched-for team this season."""
    return (
        data.groupby(["pitcher", "player_name"], dropna=False)["pitcher_team"]
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
        .rename(columns={"pitcher_team": "Team"})
    )


def calculate_playing_time(data: pd.DataFrame) -> pd.DataFrame:
    """Derive innings pitched and games started directly from pitch-level data."""

    outs = (
        data.groupby(["pitcher", "player_name"], dropna=False)["outs_recorded"]
        .sum()
        .reset_index()
        .rename(columns={"outs_recorded": "total_outs"})
    )

    # Traditional box-score notation: whole innings + tenths for extra outs (.1/.2).
    outs["IP"] = (outs["total_outs"] // 3) + (outs["total_outs"] % 3) / 10
    # True decimal value, used for threshold comparisons (e.g. 80+ IP).
    outs["IP_decimal"] = outs["total_outs"] / 3

    first_pitch_of_half_inning = (
        data.loc[data["inning"].eq(1)]
        .sort_values(["game_pk", "inning_topbot", "at_bat_number", "pitch_number"])
        .groupby(["game_pk", "inning_topbot"], as_index=False)
        .first()
    )

    starts = (
        first_pitch_of_half_inning.groupby(["pitcher", "player_name"], dropna=False)[
            "game_pk"
        ]
        .nunique()
        .reset_index()
        .rename(columns={"game_pk": "GS"})
    )

    playing_time = outs.merge(
        starts, on=["pitcher", "player_name"], how="left"
    )
    playing_time["GS"] = playing_time["GS"].fillna(0).astype(int)

    return playing_time[["pitcher", "player_name", "IP", "IP_decimal", "GS"]]


def add_pds_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_directions = {
        "K%": 1,
        "SwStr%": 1,
        "CSW%": 1,
        "BB%": -1,
        "F-Strike%": 1,
        "Chase%": 1,
        "xwOBA": -1,
        "Barrel%": -1,
        "GB%": 1,
    }

    is_starter = metrics["GS"] > 0
    meets_threshold = np.where(
        is_starter,
        metrics["batters_faced"] >= MIN_BATTERS_FACED,
        metrics["batters_faced"] >= MIN_RELIEVER_BATTERS_FACED,
    )
    qualified = metrics.loc[meets_threshold].copy()

    if qualified.empty:
        raise RuntimeError(
            "No pitchers met the minimum batters-faced requirement."
        )

    for metric, direction in metric_directions.items():
        mean = qualified[metric].mean()
        population_sd = qualified[metric].std(ddof=0)

        qualified[f"{metric} Mean"] = mean
        qualified[f"{metric} SD"] = population_sd

        if pd.isna(population_sd) or population_sd == 0:
            qualified[f"z_{metric}"] = np.nan
        else:
            qualified[f"z_{metric}"] = (
                direction * (qualified[metric] - mean) / population_sd
            )

        # Prevent one extreme result from overpowering the entire score.
        qualified[f"z_{metric}"] = qualified[f"z_{metric}"].clip(-3, 3)

    qualified["Missing Bats"] = qualified[
        ["z_K%", "z_SwStr%", "z_CSW%"]
    ].mean(axis=1)

    qualified["Command"] = qualified[
        ["z_BB%", "z_F-Strike%", "z_Chase%"]
    ].mean(axis=1)

    qualified["Contact Management"] = qualified[
        ["z_xwOBA", "z_Barrel%", "z_GB%"]
    ].mean(axis=1)

    qualified["Raw PDS"] = qualified[
        ["Missing Bats", "Command", "Contact Management"]
    ].mean(axis=1)

    qualified["PDS+"] = 100 + (10 * qualified["Raw PDS"])

    qualified["PDS+"] = qualified["PDS+"].round(1)
    qualified["Last Updated"] = END_DATE

    qualified = qualified.sort_values("PDS+", ascending=False)

    return qualified




def write_html_leaderboard(
    leaderboard: pd.DataFrame,
    template_path: Path = Path("leaderboard_template.html"),
    output_path: Path = Path("pds_leaderboard.html"),
) -> None:
    """Regenerate the shareable HTML leaderboard with this run's data baked in."""
    if not template_path.exists():
        print(f"Skipping HTML generation: {template_path} not found.")
        return

    rows = leaderboard[["player_name", "Team", "GS", "PDS+"]].copy()
    rows["Role"] = np.where(rows["GS"] > 0, "SP", "RP")

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    for _, row in rows.iterrows():
        writer.writerow([row["player_name"], row["Team"], row["Role"], row["PDS+"]])

    csv_text = buffer.getvalue().strip()

    template = template_path.read_text(encoding="utf-8")
    filled = template.replace("__PDS_DATA_PLACEHOLDER__", csv_text)

    output_path.write_text(filled, encoding="utf-8")
    print(f"Saved {output_path.resolve()}")


def main() -> None:
    raw_data = download_statcast_data()
    prepared_data = prepare_pitch_data(raw_data)
    pitcher_metrics = calculate_pitcher_metrics(prepared_data)
    playing_time = calculate_playing_time(prepared_data)

    pitcher_metrics = pitcher_metrics.merge(
        playing_time, on=["pitcher", "player_name"], how="left"
    )

    teams = calculate_teams(prepared_data)
    pitcher_metrics = pitcher_metrics.merge(
        teams, on=["pitcher", "player_name"], how="left"
    )

    leaderboard = add_pds_scores(pitcher_metrics)

    output_columns = [
        "pitcher",
        "player_name",
        "batters_faced",
        "pitches",
        "IP",
        "GS",
        "Team",
        "K%",
        "SwStr%",
        "CSW%",
        "BB%",
        "F-Strike%",
        "Chase%",
        "xwOBA",
        "Barrel%",
        "GB%",
        "Missing Bats",
        "Command",
        "Contact Management",
        "Raw PDS",
        "PDS+",
        "Last Updated",
    ]

    leaderboard[output_columns].to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(leaderboard)} pitchers to {OUTPUT_FILE.resolve()}")

    write_html_leaderboard(leaderboard)

    is_qualifying_starter = (leaderboard["GS"] > 0) & (
        leaderboard["IP_decimal"] >= MIN_STARTER_IP
    )
    is_rasmussen = (
        leaderboard["player_name"]
        .astype(str)
        .str.contains("Rasmussen", case=False, na=False)
    )

    highlighted = leaderboard.loc[is_qualifying_starter | is_rasmussen].copy()

    if not highlighted.empty:
        print(f"\nDrew Rasmussen + starters with {MIN_STARTER_IP}+ IP:")
        print(
            highlighted[
                [
                    "player_name",
                    "IP",
                    "GS",
                    "Missing Bats",
                    "Command",
                    "Contact Management",
                    "Raw PDS",
                    "PDS+",
                ]
            ].to_string(index=False)
        )
    else:
        print(
            f"\nNo pitchers matched Drew Rasmussen or the {MIN_STARTER_IP}+ IP starter threshold "
            "in the qualified leaderboard."
        )


if __name__ == "__main__":
    main()
