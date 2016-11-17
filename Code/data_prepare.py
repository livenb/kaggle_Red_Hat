import pandas as pd


def trans_char_people(df):
    for i in xrange(1, 10):
        df['char_' + str(i)] = df['char_' + str(i)].apply(lambda x: x.split()[1])
    for i in xrange(10, 38):
        df['char_' + str(i)] = df['char_' + str(i)].astype(int)
    df['group_1'] = df['group_1'].apply(lambda x: x.split()[1])
    min_day = df['date'].min()
    df['time'] = df['date'].apply(lambda x: (x-min_day).days)
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day'] = df['date'].dt.day
    df = df.drop(['date'], axis=1)
    return df


def trans_char_activity(df):
    for i in xrange(1, 11):
        df['char_' + str(i)] = df['char_' + str(i)].fillna('type -1')
        df['char_' + str(i)] = df['char_' + str(i)].apply(lambda x: x.split()[1])
    df['activity_category'] = df['activity_category'].apply(lambda x: x.split()[1])
    min_day = df['date'].min()
    df['time'] = df['date'].apply(lambda x: (x-min_day).days)
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day'] = df['date'].dt.day
    df = df.drop(['date'], axis=1)
    return df


def combine_data(p_file, a_file):
    print 'loading people'
    people = pd.read_csv(p_file, parse_dates=['date'])
    print 'loading activity'
    activity = pd.read_csv(a_file, parse_dates=['date'])
    print 'modifying people'
    people = trans_char_people(people)
    print 'modifying activity'
    activity = trans_char_activity(activity)
    print 'merging'
    new_df = activity.merge(people,
                            left_on='people_id',
                            right_on='people_id',
                            suffixes=['_a', '_p'])
    # new_df['days_diff'] = (new_df['date_a'] - new_df['date_p']).apply(lambda x: x.days)
    return new_df


if __name__ == '__main__':
    p_file = '../Data/people.csv'
    a_train_file = '../Data/act_train.csv'
    a_test_file = '../Data/act_train.csv'
    print 'train file'
    train_df = combine_data(p_file, a_train_file)
    print 'test file'
    test_df = combine_data(p_file, a_test_file)
    print 'writing'
    train_df.to_csv('../Data/train.csv', index=False)
    test_df.to_csv('../Data/test.csv', index=False)
