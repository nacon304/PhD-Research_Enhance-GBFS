from copy_of_js_ms import Copy_of_js_ms
import os
import gbfs_globals as GG

if __name__ == "__main__":
    # ví dụ:
    mat_dir = "D:/PhD/The First Paper/Code Implement/GBFS-SND/Evaluation/Result Implementation 3"
    GG.visual_dir = "D:/PhD/The First Paper/Code Implement/GBFS-SND/Evaluation/Visualize Implementation 3"
    os.makedirs(mat_dir, exist_ok=True)

    product, T, datasetName, assiNum = Copy_of_js_ms(
        dataIdx=1,
        delt=0.3,
        omega=0.5,
        RUNS=30
    )
    print(datasetName)
    print(product[-2:])  # mean/std
