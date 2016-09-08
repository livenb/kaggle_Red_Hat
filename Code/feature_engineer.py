import pandas as pd


def combine_data(p_file, a_file):
    people = pd.read_csv(p_file, parse_dates=['date'])
    activity = pd.read_csv(a_file, parse_dates=['date'])
    new_df = activity.merge(people, left_on='people_id', right_on='people_id', suffixes=['_a', '_p'])
    return new_df


if __name__ == '__main__':
    p_file = '../Data/people.csv'
    a_train_file = '../Data/act_train.csv'
    a_test_file = '../Data/act_train.csv'
    train_df = combine_data(p_file, a_train_file)
    test_df = combine_data(p_file, a_test_file)
