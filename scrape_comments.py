import numpy as np
import pandas as pd

import praw
import urllib.request as url
from tqdm import tqdm, tqdm_notebook

# add client_id and client_secret here
reddit = praw.Reddit(
	client_id = '',
    client_secret= '',
    user_agent='my user agent',
    username= '',
    password= '',
)



subreddit = reddit.subreddit('formcheck')


def fetch_subreddit_info(limit=15):
	"""
	This function is used to fetch subreddit comments from formcheck
	"""
	meta_data_list = []
	for submission in tqdm(subreddit.hot(limit=limit)):
	    if submission.is_video:
	        meta_data = submission.media['reddit_video']
	        meta_data['title'] = submission.title
	        meta_data['id'] = submission.id
	        meta_data['upvote'] = submission.score
	        ls = []
	        for comment in submission.comments:
	            first_comment = comment.body
	            comment_id = comment.id
	            replies = []
	            for r in comment.replies.list():
	                replies.append(r.body)
	            ls.append({
	                'first_comment': first_comment,
	                'comment_id': comment_id,
	                'subreddit': comment.subreddit.display_name,
	                'subreddit_id': comment.subreddit_id,
	                'upvote': comment.score,
	                'replies': replies
	            })
	        meta_data['comments'] = ls
	        meta_data_list.append(meta_data)
	return meta_data_list


if __name__ == '__main__':
	meta_data_list = fetch_subreddit_info(limit=30)
	# TO DO: save data
	print(meta_data_list)