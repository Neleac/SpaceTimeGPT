import argparse
import json

def convert_vatex_to_cider(input_path, output_path):
    cider_data = []
    with open(input_path) as vatex_file:
        vatex_data = json.load(vatex_file)

        for datum in vatex_data:
            video_id, captions = datum['videoID'], datum['enCap']

            for caption in captions:
                cider_data.append({'image_id': video_id, 'caption': caption})

    cider_json = json.dumps(cider_data)
    with open(output_path, 'w') as cider_file:
        cider_file.write(cider_json)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert VaTeX captions to CIDEr format')
    parser.add_argument('-i', '--input', required=True, help='path to input VaTeX captions')
    parser.add_argument('-o', '--output', required=True, help='path to output CIDEr captions')
    args = parser.parse_args()

    convert_vatex_to_cider(args.input, args.output)
