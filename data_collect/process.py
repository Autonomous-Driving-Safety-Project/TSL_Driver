import pandas as pd
import numpy as np
import sparse
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

PROCESS_NUM = 12

def discretize(df):

    relate_speed_bins = [-5, 0, 5, 10, np.nan]
    back_relate_position_x_bins = [-200, -100, np.nan]
    front_relate_position_x_bins = [100, 200, np.nan]

    df["dis_surrounds_LF_relate_speed"] = np.digitize(
        df["surrounds_LF_relate_speed"], bins=relate_speed_bins
    )
    df["dis_surrounds_LB_relate_speed"] = np.digitize(
        df["surrounds_LB_relate_speed"], bins=relate_speed_bins
    )
    df["dis_surrounds_F_relate_speed"] = np.digitize(
        df["surrounds_F_relate_speed"], bins=relate_speed_bins
    )
    df["dis_surrounds_B_relate_speed"] = np.digitize(
        df["surrounds_B_relate_speed"], bins=relate_speed_bins
    )
    df["dis_surrounds_RF_relate_speed"] = np.digitize(
        df["surrounds_RF_relate_speed"], bins=relate_speed_bins
    )
    df["dis_surrounds_RB_relate_speed"] = np.digitize(
        df["surrounds_RB_relate_speed"], bins=relate_speed_bins
    )

    df["dis_surrounds_LF_relate_position_x"] = np.digitize(
        df["surrounds_LF_relate_position_x"], bins=front_relate_position_x_bins
    )
    df["dis_surrounds_F_relate_position_x"] = np.digitize(
        df["surrounds_F_relate_position_x"], bins=front_relate_position_x_bins
    )
    df["dis_surrounds_RF_relate_position_x"] = np.digitize(
        df["surrounds_RF_relate_position_x"], bins=front_relate_position_x_bins
    )

    df["dis_surrounds_LB_relate_position_x"] = np.digitize(
        df["surrounds_LB_relate_position_x"], bins=back_relate_position_x_bins
    )
    df["dis_surrounds_B_relate_position_x"] = np.digitize(
        df["surrounds_B_relate_position_x"], bins=back_relate_position_x_bins
    )
    df["dis_surrounds_RB_relate_position_x"] = np.digitize(
        df["surrounds_RB_relate_position_x"], bins=back_relate_position_x_bins
    )

    return df


def process_episode(df, dims):
    stat = sparse.DOK(dims, dtype=np.int32)
    for frame in df["frame"].unique():
        for vehicle_id in df[df["frame"] == frame]["vehicle_id"].unique():
            this_frame = df[
                (df["frame"] == frame) & (df["vehicle_id"] == vehicle_id)
            ].iloc[0]
            future_frames = df[(df["frame"] > frame)]
            if len(future_frames) == 0:
                continue
            for next_frame in future_frames["frame"]:
                stat[
                    int(this_frame["dis_surrounds_LF_relate_speed"]),
                    int(this_frame["dis_surrounds_LB_relate_speed"]),
                    int(this_frame["dis_surrounds_F_relate_speed"]),
                    int(this_frame["dis_surrounds_B_relate_speed"]),
                    int(this_frame["dis_surrounds_RF_relate_speed"]),
                    int(this_frame["dis_surrounds_RB_relate_speed"]),
                    int(this_frame["dis_surrounds_LF_relate_position_x"]),
                    int(this_frame["dis_surrounds_LB_relate_position_x"]),
                    int(this_frame["dis_surrounds_F_relate_position_x"]),
                    int(this_frame["dis_surrounds_B_relate_position_x"]),
                    int(this_frame["dis_surrounds_RF_relate_position_x"]),
                    int(this_frame["dis_surrounds_RB_relate_position_x"]),
                    int(next_frame - frame),
                    int(this_frame["relation_with_ego"]),
                    int(
                        df[
                            (df["frame"] == next_frame)
                            & (df["vehicle_id"] == vehicle_id)
                        ].iloc[0]["relation_with_ego"] if len(df[(df["frame"] == next_frame) & (df["vehicle_id"] == vehicle_id)]) > 0 else 0
                    ),
                ] += 1
    return stat


def partial_sum(results, start, end, dims):
    partial_stat = sparse.DOK(dims, dtype=np.int32)
    for i, result in enumerate(results[start:end]):
        print(f"Summing from {start} to {end-1}: {i}/{end-start}")
        partial_stat += result
    return partial_stat

def process(df, save_path):

    print("Discretizing index...")
    df = discretize(df)

    dims = []
    for col in [
        "dis_surrounds_LF_relate_speed",
        "dis_surrounds_LB_relate_speed",
        "dis_surrounds_F_relate_speed",
        "dis_surrounds_B_relate_speed",
        "dis_surrounds_RF_relate_speed",
        "dis_surrounds_RB_relate_speed",
        "dis_surrounds_LF_relate_position_x",
        "dis_surrounds_LB_relate_position_x",
        "dis_surrounds_F_relate_position_x",
        "dis_surrounds_B_relate_position_x",
        "dis_surrounds_RF_relate_position_x",
        "dis_surrounds_RB_relate_position_x",
        "frame",
        "relation_with_ego",
    ]:
        dims.append(df[col].max() + 1)
        if col == "relation_with_ego":
            dims.append(df[col].max() + 1)
    dims = tuple(dims)

    process_episode_partial = partial(process_episode, dims=dims)
    
    

    episode_num = df["episode"].max() + 1

    print(f"{episode_num} episodes found. Begin processing...")

    # for episode in tqdm(range(episode_num)):
    #     stat += process_episode(df[df['episode']==episode], dims)
    
    with Pool(processes=PROCESS_NUM) as pool:
        # results = tqdm(pool.starmap(process_episode, [(df[df['episode']==episode], dims) for episode in range(episode_num)]), total=episode_num)
        results = list(tqdm(pool.imap_unordered(process_episode_partial, [df[df['episode']==episode] for episode in range(episode_num)]), total=episode_num))
    
    print("Summing...")
    # stat = sparse.DOK(dims, dtype=np.int32)
    # for result in tqdm(results):
    #     stat += result
    results_per_thread = len(results) // PROCESS_NUM
    with Pool(processes=PROCESS_NUM) as pool:
        partial_results = pool.starmap(partial_sum, [(results, i * results_per_thread, (i+1) * results_per_thread if i < PROCESS_NUM - 1 else len(results), dims) for i in range(PROCESS_NUM)])
    print("Merging...")
    stat = sparse.DOK(dims, dtype=np.int32)
    for result in tqdm(partial_results):
        stat += result
    print(stat)
    print("Done. Saving...")
    sparse.save_npz(save_path, stat.to_coo())
    print("Process successfully finished.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print("Reading data...")
    df = pd.read_csv(args.input)
    if df is None:
        print("No data found.")
        return
    process(df, args.output)


if __name__ == "__main__":
    main()
