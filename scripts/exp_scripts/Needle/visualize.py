
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import glob
import os

SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
RESULTS_DIR = os.path.join(SOURCE_DIR, "results_needle", "results")
SAVE_PATH = os.path.join(SOURCE_DIR, "results_needle", "img")
MODEL_NAME = "LlAMA3"
PRETRAINED_LEN=33000


def draw_one_folder(folder_path):
    run_name = os.path.basename(os.path.normpath(folder_path))
    model_name = run_name if run_name else MODEL_NAME
    print("model_name = %s" % model_name)

    # Using glob to find all json files in the directory
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    if len(json_files) == 0:
        print("No json found in %s, skip." % folder_path)
        return
    # import ipdb; ipdb.set_trace()

    # List to hold the data
    data = []

    # Iterating through each file and extract the 3 columns we need
    for file in json_files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            # Extracting the required fields
            
            try:
                document_depth = json_data.get("depth_percent", None)
                context_length = json_data.get("context_length", None)
            except:
                import pdb
                pdb.set_trace()
            # score = json_data.get("score", None)
            model_response = json_data.get("model_response", None).lower()
            needle = json_data.get("needle", None).lower()
            expected_answer = "eat a sandwich and sit in Dolores Park on a sunny day.".lower().split()
            score = len(set(model_response.split()).intersection(set(expected_answer))) / len(set(expected_answer))
            # Appending to the list
            data.append({
                "Document Depth": document_depth,
                "Context Length": context_length,
                "Score": score
            })

    # Creating a DataFrame
    df = pd.DataFrame(data)

    locations = list(df["Context Length"].unique())
    locations.sort()
    for li, l in enumerate(locations):
        if(l > PRETRAINED_LEN): break
    pretrained_len = li

    print(df.head())
    overall_acc = df["Score"].mean()
    print("Overall score %.3f" % overall_acc)

    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
    pivot_table.iloc[:5, :5]

    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with better aesthetics
    f = plt.figure(figsize=(38, 8))  # Can adjust these dimensions as needed
    heatmap = sns.heatmap(
        pivot_table,
        vmin=0, vmax=1,
        cmap=cmap,
        cbar_kws={'label': 'Score'},
        linewidths=0.5,  # Adjust the thickness of the grid lines here
        linecolor='grey',  # Set the color of the grid lines
        linestyle='--'
    )

    # Put overall accuracy at the top-right corner of the figure for quick reading.
    heatmap.text(
        0.98, 0.98,
        f"ACC: {overall_acc:.3f}",
        transform=heatmap.transAxes,
        ha='right',
        va='top',
        fontsize=18,
        color='white',
        bbox=dict(facecolor='black', alpha=0.45, edgecolor='none', boxstyle='round,pad=0.3')
    )


    # More aesthetics
    title_name = model_name if model_name else MODEL_NAME
    plt.title(f'Pressure Testing {title_name} \nFact Retrieval Across Context Lengths ("Needle In A HayStack")', fontsize=18)  # Adds a title
    plt.xlabel('Token Limit', fontsize=18)  # X-axis label
    plt.ylabel('Depth Percent', fontsize=18)  # Y-axis label
    plt.xticks(rotation=45, fontsize=18)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0, fontsize=18)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area

    # Add a vertical line at the desired column index
    plt.axvline(x=pretrained_len + 0.8, color='white', linestyle='--', linewidth=4)

    save_path = os.path.join(SAVE_PATH, "%s.png" % model_name)
    print("saving at %s" % save_path)
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    # Read experiment outputs from <source_dir>/results_needle/results
    results_root = RESULTS_DIR
    os.makedirs(SAVE_PATH, exist_ok=True)
    print("Source dir: %s" % SOURCE_DIR)
    print("Results dir: %s" % results_root)
    print("Save dir: %s" % SAVE_PATH)
    subfolders = [p for p in glob.glob(os.path.join(results_root, "*/")) if os.path.isdir(p)]
    subfolders.sort()

    if len(subfolders) == 0:
        print("No result subfolders found under %s" % results_root)
        return

    print("Found %d result folders." % len(subfolders))
    for folder_path in subfolders:
        print("\nProcessing folder: %s" % folder_path)
        draw_one_folder(folder_path)


if __name__ == "__main__":
    main()
