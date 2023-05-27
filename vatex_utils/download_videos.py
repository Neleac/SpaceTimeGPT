import argparse
import json
import subprocess


def download_vatex_videos(json_path, output_path):
    with open(json_path) as json_file:
        json_data = json.load(json_file)
        
        for data in json_data:
            video_id = data['videoID']
            video_id_tokens = video_id.split('_')
            youtube_id = '_'.join(video_id_tokens[:-2])
            start_time, end_time = video_id_tokens[-2], video_id_tokens[-1]

            # download video (lowest res greater than 224p, no DASH)
            cmd = 'yt-dlp -f "wv*[height>=224][protocol!*=dash]/bv*[protocol!*=dash]" --download-sections "*%s-%s" -o "%s%s.%%(ext)s" "https://www.youtube.com/watch?v=%s"' \
                % (start_time, end_time, output_path, video_id, youtube_id)

            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError:   # video no longer available
                continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='download VaTeX dataset YouTube videos')
    parser.add_argument('-i', '--input', required=True, help='path to input VaTeX dataset file')
    parser.add_argument('-o', '--output', required=True, help='directory in which the videos will be downloaded')
    args = parser.parse_args()

    download_vatex_videos(args.input, args.output)
