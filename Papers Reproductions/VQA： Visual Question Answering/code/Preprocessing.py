import operator
import argparse
import json
import spacy


def get_model_answer(answers):
    candidates = {}
    for i in range(10):
        candidates[answers[i]['answer']] = 1

    for i in range(10):
        candidates[answers[i]['answer']] += 1

    return max(candidates.items(), key=operator.itemgetter(1))[0]


def get_all_answer(answers):
    answer_list = []
    for i in range(10):
        answer_list.append(answers[i]['answer'])

    return ';'.join(answer_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-split', type=str, default='train',
                        help='Specify which part of the dataset you want to dump to text. '
                             'Your options are: train, val, test, test-dev')
    parser.add_argument('-answers', type=str, default='modal',
                        help='Specify if you want to dump just the most frequent answer for each questions (modal), '
                             'or all the answers (all)')
    args = parser.parse_args()

    nlp = spacy.load('en')  # used for counting number of tokens

    if args.split == 'train':
        annotation_file = '../mscoco_train2014_annotations.json'
        question_file = '../OpenEnded_mscoco_train2014_questions.json'
        questions_file = open('..//questions_train2014.txt', 'wb')
        questions_id_file = open('../questions_id_train2014.txt', 'wb')
        questions_lengths_file = open('../questions_lengths_train2014.txt', 'wb')

        if args.answers == 'modal':
            answers_file = open('../answers_train2014_modal.txt', 'wb')
        elif args.answers == 'all':
            answers_file = open('../answers_train2014_all.txt', 'wb')

        coco_image_id = open('../images_train2014.txt', 'wb')
        data_split = 'training data'

    elif args.split == 'val':
        annotation_file = '../mscoco_val2014_annotations.json'
        question_file = '../OpenEnded_mscoco_val2014_questions.json'
        questions_file = open('../questions_val2014.txt', 'wb')
        questions_id_file = open('../questions_id_val2014.txt', 'wb')
        questions_lengths_file = open('../questions_lengths_val2014.txt', 'wb')

        if args.answers == 'modal':
            answers_file = open('../answers_val2014_modal.txt', 'wb')
        elif args.answers == 'all':
            answers_file = open('../answers_val2014_all.txt', 'wb')

        coco_image_id = open('../images_val2014_all.txt', 'wb')
        data_split = 'validation data'

    elif args.split == 'test-dev':
        question_file = '../OpenEnded_mscoco_test-dev2015_questions.json'
        questions_file = open('../questions_test-dev2015.txt', 'wb')
        questions_id_file = open('../questions_id_test-dev2015.txt', 'wb')
        questions_lengths_file = open('../questions_lengths_test-dev2015.txt', 'wb')

        coco_image_id = open('../images_test-dev2015.txt', 'wb')
        data_split = 'test-dev data'

    elif args.split == 'test':
        question_file = '../OpenEnded_mscoco_test2015_questions.json'
        questions_file = open('../questions_test2015.txt', 'wb')
        questions_id_file = open('../questions_id_test2015.txt', 'wb')
        questions_lengths_file = open('../questions_lengths_test2015.txt', 'wb')

        coco_image_id = open('../images_test2015.txt', 'wb')
        data_split = 'test data'

    else:
        raise RuntimeError('Incorrect split. Your choices are:\n train \n val \n test-dev \n test')

    # initialize VQA api for QA annotations
    # vqa=VQA(annFile, quesFile)

    questions = json.load(open(question_file, 'r'))
    question_matrix = questions['questions']
    if args.split == 'train' or args.split == 'val':
        question_answer = json.load(open(annotation_file, 'r'))
        question_answer = question_answer['annotations']

    print('Dumping questions, answers, questionIDs, imageIDs, and questions lengths to text files...')
    for i, q in zip(range(len(question_matrix)), question_matrix):
        questions_file.write((q['question'] + '\n').encode('utf8'))
        questions_lengths_file.write((str(len(nlp(q['question']))) + '\n').encode('utf8'))
        questions_id_file.write((str(q['question_id']) + '\n').encode('utf8'))
        coco_image_id.write((str(q['image_id']) + '\n').encode('utf8'))

        if args.split == 'train' or args.split == 'val':
            if args.answers == 'modal':
                answers_file.write(get_model_answer(question_answer[i]['answers']).encode('utf8'))
            elif args.answers == 'all':
                answers_file.write(get_all_answer(question_answer[i]['answers']).encode('utf8'))
            answers_file.write('\n'.encode('utf8'))

    print('completed dumping', data_split)


if __name__ == "__main__":
    main()

