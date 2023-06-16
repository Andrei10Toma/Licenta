import json
from world import *


def remove_texts_from_reviews():
    with open(YELP_REVIEW, 'r') as f:
        with open(os.path.join('dataset', 'yelp', 'yelp_academic_dataset_review_no_text.json'), 'w') as g:
            for _, line in enumerate(f):
                data = json.loads(line)
                data.pop('text')
                g.write(json.dumps(data) + os.linesep)


if __name__ == '__main__':
    remove_texts_from_reviews()
