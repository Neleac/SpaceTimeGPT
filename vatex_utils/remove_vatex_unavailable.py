import argparse
import json
import os


def clean_vatex_file(videos_path, file_path):
    video_ids = set()
    for _, _, files in os.walk(videos_path):
        for file in files:
            filename, _ = os.path.splitext(file)
            video_ids.add(filename)

    clean_data = []
    with open(file_path) as json_file:
        json_data = json.load(json_file)

        for data in json_data:
            video_id = data['videoID']
            if video_id in video_ids:
                clean_data.append(data)

    clean_json = json.dumps(clean_data)
    with open(file_path, 'w') as json_file:
        json_file.write(clean_json)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='remove unavailable videos from VaTeX json file, based on ones downloaded')
    parser.add_argument('-v', '--videos', required=True, help='path to downloaded VaTeX dataset videos')
    parser.add_argument('-f', '--file', required=True, help='path to VaTeX dataset json file')
    args = parser.parse_args()

    clean_vatex_file(args.videos, args.file)