from sqlalchemy import MetaData, Table, create_engine, select
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# https://docs.sqlalchemy.org/en/20/core/reflection.html
engine = create_engine(DATABASE_URL)
metadata_obj = MetaData()
metadata_obj.reflect(bind=engine)

# Load the tables
Attempt = Table("Attempt", metadata_obj, autoload_with=engine)
Foul = Table("Foul", metadata_obj, autoload_with=engine)
Game = Table("Game", metadata_obj, autoload_with=engine)
Gameday = Table("Gameday", metadata_obj, autoload_with=engine)
Player = Table("Player", metadata_obj, autoload_with=engine)
PlayersInGameday = Table(
    "PlayersInGameday", metadata_obj, autoload_with=engine)
PlayersInTeam = Table("PlayersInTeam", metadata_obj, autoload_with=engine)
Team = Table("Team", metadata_obj, autoload_with=engine)


def execute_query_to_df(sql):
    # Execute the query and retrieve the results into a DataFrame
    with engine.connect() as conn:
        result = conn.execute(sql)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

    return df


def get_stats(userId, gamedayId):
    stmt = select(Gameday.c.id).where(Gameday.c.userId == userId)
    gamedays_df = execute_query_to_df(stmt)
    gamedays_ids = [
        int(gamedayId)] if gamedayId else gamedays_df['id'].tolist()

    stmt = select(Game).where(Game.c.gamedayId.in_(gamedays_ids))
    games_df = execute_query_to_df(stmt)

    # Get id's of all teams that are in the selected Gamedays. Also count matches played by each team.
    allTeamIds = pd.concat([games_df["teamAId"], games_df["teamBId"]])
    allTeamIdsUniq, MPs = np.unique(allTeamIds, return_counts=True)

    # Get all teams that are in the selected Gamedays
    stmt = select(Team.c.id, Team.c.name).where(
        Team.c.id.in_(allTeamIdsUniq.tolist()))
    teams_df = execute_query_to_df(stmt)

    # Determine winners, losers, and draws
    # Pandas' apply function is used to determine the winner and loser of each game in one operation.
    games_df['winnerId'] = games_df.apply(lambda row: row['teamAId'] if row['scoreTeamA'] > row['scoreTeamB'] else (
        row['teamBId'] if row['scoreTeamB'] > row['scoreTeamA'] else 'draw'), axis=1)
    games_df['loserId'] = games_df.apply(lambda row: row['teamBId'] if row['scoreTeamA'] > row['scoreTeamB'] else (
        row['teamAId'] if row['scoreTeamB'] > row['scoreTeamA'] else 'draw'), axis=1)

    winners = games_df['winnerId'].value_counts().to_dict()
    losers = games_df['loserId'].value_counts().to_dict()

    Ws = [winners.get(tId, 0) for tId in allTeamIdsUniq]
    Ls = [losers.get(tId, 0) for tId in allTeamIdsUniq]
    Ds = [MPs[i] - Ws[i] - Ls[i] for i in range(len(allTeamIdsUniq))]

    dfTeamOverall = pd.DataFrame({
        "TeamID": allTeamIdsUniq,
        "MP": MPs,
        "W": Ws,
        "D": Ds,
        "L": Ls,
    })

    dfTeamOverall["Pts"] = dfTeamOverall["W"]*3 + dfTeamOverall["D"]
    dfTeamOverall["Pts/MP"] = (dfTeamOverall["Pts"] /
                               dfTeamOverall["MP"]).round(2)

    dfTeamOverall.set_index('TeamID', inplace=True)

    gf_teamA = games_df.groupby('teamAId')['scoreTeamA'].sum()
    gf_teamB = games_df.groupby('teamBId')['scoreTeamB'].sum()
    GFs = gf_teamA.add(gf_teamB, fill_value=0)

    ga_teamA = games_df.groupby('teamAId')['scoreTeamB'].sum()
    ga_teamB = games_df.groupby('teamBId')['scoreTeamA'].sum()
    GAs = ga_teamA.add(ga_teamB, fill_value=0)

    # Add to the DataFrame
    dfTeamOverall = dfTeamOverall.join(GFs.rename('GF')).join(GAs.rename('GA'))

    dfTeamOverall["GF/MP"] = (dfTeamOverall["GF"] /
                              dfTeamOverall["MP"]).astype(float).round(2)
    dfTeamOverall["GA/MP"] = (dfTeamOverall["GA"] /
                              dfTeamOverall["MP"]).astype(float).round(2)

    # Merge with teams_df to get the team names
    dfTeamOverall = pd.merge(dfTeamOverall, teams_df,
                             left_on='TeamID', right_on='id', how='left')
    dfTeamOverall.rename(columns={'name': 'Team'}, inplace=True)
    # dfTeamOverall.drop(columns=['id'], inplace=True)

    # Sort and rank based on points
    dfTeamOverall = dfTeamOverall.sort_values("Pts", ascending=False)
    dfTeamOverall.index = np.arange(len(dfTeamOverall.index))
    dfTeamOverall["Rank"] = dfTeamOverall.index + 1

    # Teams table
    stmt = select(Player.c.name, Player.c.id).where(Player.c.userId == userId)
    players_df = execute_query_to_df(stmt)
    stmt = select(PlayersInTeam).where(
        PlayersInTeam.c.playerId.in_(players_df["id"].tolist()))
    players_in_team_df = execute_query_to_df(stmt)

    dfPlayerTeamStats = pd.merge(
        players_in_team_df, dfTeamOverall, left_on='teamId', right_on='id', how='left')

    dfPlayerOverall = dfPlayerTeamStats.groupby(
        'playerId').sum(numeric_only=True).reset_index()
    dfPlayerOverall = pd.merge(
        dfPlayerOverall, players_df, left_on='playerId', right_on='id', how='left')
    dfPlayerOverall.drop(
        columns=["teamId", "id_x", "id_y", "Rank"], inplace=True)
    dfPlayerOverall.rename(columns={'name': 'Player'}, inplace=True)

    dfPlayerOverall["GF/MP"] = np.where(dfPlayerOverall["MP"] != 0,
                                        (dfPlayerOverall["GF"]/dfPlayerOverall["MP"]).astype(float).round(2), 0)
    dfPlayerOverall["GA/MP"] = np.where(dfPlayerOverall["MP"] != 0,
                                        (dfPlayerOverall["GA"]/dfPlayerOverall["MP"]).astype(float).round(2), 0)

    # Shots table
    stmt = select(Attempt).where(Attempt.c.gameId.in_(games_df['id'].tolist()))
    attempts_df = execute_query_to_df(stmt)

    dfShots = dfPlayerOverall[["Player", "playerId", "MP"]].copy()

    # Filter the attempts for non-penalty shots
    non_penalty_attempts = attempts_df[~attempts_df['penalty']]
    # Count the shots for each player
    shots = non_penalty_attempts.groupby('shooterId').size()
    dfShots = dfShots.join(shots.rename('S'), on='playerId')

    goals_df = attempts_df[(attempts_df['penalty'] == False)
                           & (attempts_df['goal'] == True)]
    goals = goals_df.groupby('shooterId').size()
    dfShots = dfShots.join(goals.rename('G'), on='playerId')

    sots = non_penalty_attempts[non_penalty_attempts['onTarget'] == True].groupby(
        'shooterId').size()
    dfShots = dfShots.join(sots.rename('SoT'), on='playerId')
    dfShots["SoT/MP"] = (dfShots["SoT"] / dfShots["MP"]).round(2)

    assists = non_penalty_attempts[non_penalty_attempts['goal'] == True].groupby(
        'assistedId').size()
    dfShots = dfShots.join(assists.rename('A'), on='playerId')

    dfShots["G/MP"] = (dfShots["G"] / dfShots["MP"]).round(2)
    dfShots["A/MP"] = (dfShots["A"] / dfShots["MP"]).round(2)
    dfShots["S/MP"] = (dfShots["S"] / dfShots["MP"]).round(2)
    dfShots["S/G"] = (dfShots["S"] / dfShots["G"]).round(2)

    # Conversion Rate
    dfShots['Conversion Rate'] = (dfShots['G'] / dfShots['SoT']).round(2)

    # Assist to Goal Ratio
    dfShots['Assist to Goal Ratio'] = (dfShots['A'] / dfShots['G']).round(2)

    # Shot Accuracy
    dfShots['Shot Accuracy'] = (dfShots['SoT'] / dfShots['S']).round(2)

    # Penalty Conversion Rate
    penalty_attempts = attempts_df[attempts_df['penalty']]
    penalty_goals = penalty_attempts[penalty_attempts['goal'] == True]
    penalty_goals_count = penalty_goals.groupby('shooterId').size()
    penalty_attempts_count = penalty_attempts.groupby('shooterId').size()

    dfShots = dfShots.join(penalty_goals_count.rename(
        'Penalty Goals'), on='playerId')
    dfShots = dfShots.join(penalty_attempts_count.rename(
        'Penalty Attempts'), on='playerId')

    dfShots['Penalty Conversion Rate'] = (
        dfShots['Penalty Goals'] / dfShots['Penalty Attempts']).round(2)

    # Average Shot Distance
    avg_shot_distance = non_penalty_attempts.groupby(
        'shooterId')['distance'].mean().round(2)
    dfShots = dfShots.join(avg_shot_distance.rename(
        'AvgShotDistance'), on='playerId')

    # Goal Distance Ratio
    # adjust this value based on your definition of a long-range shot
    long_range_goals = goals_df[goals_df['distance'] > 16]
    long_range_goals_ratio = (long_range_goals.groupby(
        'shooterId').size() / goals).round(2)
    dfShots = dfShots.join(long_range_goals_ratio.rename(
        'LongRangeGoalRatio(16m+)'), on='playerId')

    def categorize_attempts_by_time(df, time_buckets):
        """
        Categorize attempts based on the time of the game. Each attempt is put into a bucket based on the provided time buckets.

        Args:
            df (pd.DataFrame): DataFrame with the attempts data.
            time_buckets (list of int): List of time thresholds to define the buckets.

        Returns:
            pd.DataFrame: Original DataFrame with an additional column for the time bucket of each attempt.
        """
        df = df.copy()  # make a copy of the DataFrame
        time_buckets = sorted(time_buckets)
        labels = [
            f"{time_buckets[i]}-{time_buckets[i+1]}" for i in range(len(time_buckets)-1)]
        df['timeBucket'] = pd.cut(
            df['time'], bins=time_buckets, labels=labels, right=False)
        return df

    # Create time buckets
    time_buckets = list(range(0, 100, 15)) + [np.inf]

    # Filter data for non-penalty attempts and goals
    non_penalty_attempts = attempts_df[~attempts_df['penalty']].copy()
    goals_df = attempts_df[(attempts_df['penalty'] == False) & (
        attempts_df['goal'] == True)].copy()

    # Categorize attempts and goals by time
    non_penalty_attempts = categorize_attempts_by_time(
        non_penalty_attempts, time_buckets)
    goals_df = categorize_attempts_by_time(goals_df, time_buckets)

    # Create distributions for attempts and goals
    attempts_distribution = non_penalty_attempts.groupby(
        ['shooterId', 'timeBucket']).size().unstack(fill_value=0)
    goals_distribution = goals_df.groupby(
        ['shooterId', 'timeBucket']).size().unstack(fill_value=0)

    # Join the distributions to dfShots
    dfShots = dfShots.join(
        attempts_distribution.add_suffix('_attempts'), on='playerId')
    dfShots = dfShots.join(
        goals_distribution.add_suffix('_goals'), on='playerId')

    # adjust this value based on your definition of a clutch goal
    clutch_goals = goals_df[goals_df['time'] >= 80]
    clutch_goals_count = clutch_goals.groupby('shooterId').size()
    dfShots = dfShots.join(clutch_goals_count.rename(
        'ClutchGoals'), on='playerId')

    dfShots.fillna(0, inplace=True)
    dfShots.replace(np.inf, 0, inplace=True)

    dfShots = dfShots.sort_values("G", ascending=False)
    dfShots.index = np.arange(len(dfShots.index))
    dfShots["Rank"] = dfShots.index + 1

    return (dfTeamOverall, dfPlayerOverall, dfShots)
