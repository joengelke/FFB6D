import argparse
import os
import glob
import re


def main():
    parser = argparse.ArgumentParser(description="Arg parser")

    parser.add_argument(
        "-datadir", type=str, default=None, help="Path to directory"
    )
    parser.add_argument(
        "-output", type=str, default=None, help="Path to directory"
    )

    args = parser.parse_args()

    with open(args.output+'/own_data_list.txt', 'w') as f:
        for subdir, dirs, files in os.walk(args.datadir):
            files.sort(key=lambda g: int(re.sub('\D', '', g)))
            i = 0
            for file in files:
                if i % 4 == 0:
                    f.write('owndata/'+os.path.basename(subdir)+'/'+file[0:10]+'\n')
                i += 1


if __name__ == "__main__":
    main()