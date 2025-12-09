import os
from scipy.io import savemat
from runn import runn


def main():
    mat_dir = "D:/PhD/The First Paper/Code Implement/GBFS-SND/Result"
    os.makedirs(mat_dir, exist_ok=True)

    # for i in range(1, 15):
    for i in range(1, 5):
        PP, datasetname = runn(i)

        rows_main = PP[:-2]
        ACC = [[row[0], row[1], row[3]] for row in rows_main]

        row_m = PP[-2]
        ACC_m = [row_m[0], row_m[1], row_m[3]]

        row_s = PP[-1]
        ACC_s = [row_s[0], row_s[1], row_s[3]]

        selectfeat = [row[2] for row in PP]

        mat_name = os.path.join(mat_dir, f"{datasetname}.mat")

        savemat(
            mat_name,
            {
                "ACC": ACC,
                "ACC_m": ACC_m,
                "ACC_s": ACC_s,
                "selectfeat": selectfeat,
            },
        )


if __name__ == "__main__":
    main()
