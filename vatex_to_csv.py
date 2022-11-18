import argparse
import csv
import json


def convert_vatex_to_csv(json_path, csv_path):
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='|')

        with open(json_path) as json_file:
            json_data = json.load(json_file)

            for data in json_data:
                video_id, captions = data['videoID'], data['enCap']
                csv_writer.writerow([video_id] + captions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert VaTeX dataset json file to csv')
    parser.add_argument('-i', '--input', required=True, help='path to VaTeX dataset json file')
    parser.add_argument('-o', '--output', required=True, help='path to output csv file')
    args = parser.parse_args()

    convert_vatex_to_csv(args.input, args.output)