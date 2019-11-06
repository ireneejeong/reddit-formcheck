import numpy as np
import pandas as pd
from itertools import chain
from collections import Counter

def count(ls):
    flatten_ls = list(chain.from_iterable(ls))
    flatten_ls = [e for e in flatten_ls if e is not '']
    return Counter(flatten_ls)

if __name__ == '__main__':

    df = pd.read_csv('formcheck_comments - formcheck_comments.csv')
    df['correction type'] = df.fillna('')['correction type'].map(lambda x: [t.strip() for t in x.split(',')])
    popular_type_df = df.groupby('type').size().reset_index().sort_values(0, ascending=False).head(3)
    problem_df = popular_type_df.merge(df.groupby('type')['correction type'].\
                                        agg(lambda x: count(x)).reset_index(), on='type')
    problem_df.rename(columns={0: 'n_type'}, inplace=True)
    correction_type_df = pd.DataFrame(list(problem_df['correction type'].map(lambda x: dict(x)).values)).fillna(0)
    correction_type_df = pd.concat((problem_df[['type', 'n_type']], correction_type_df), axis=1)

    # select popular correction types
    selected_cols = [c for c in correction_type_df.columns if c is not 'type']
    n_corrections = [correction_type_df[c].sum() for c in selected_cols]
    indices = np.argsort(n_corrections)[::-1][0:15]

    # most popular correction type
    correction_type_df[['type'] + [selected_cols[i] for i in indices]]
