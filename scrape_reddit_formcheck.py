import os
import sys
import json
import praw
import urllib.request
from tqdm import tqdm, tqdm_notebook


# add client_id and client_secret here
reddit = praw.Reddit(
    client_id='',
    client_secret='',
    user_agent='my user agent',
    username='',
    password='',
)
subreddit = reddit.subreddit('formcheck')


def get_comment_replies(comment):
    """
    Get list of replies after a given comment
    """
    try:
        replies = [c.body for c in comment.replies.list()]
    except:
        replies = []
    return replies


def fetch_subreddit_info(limit=15):
    """
    This function is used to fetch subreddit metadata and comments from subreddit formcheck
    """
    videos_metadata, videos_comment = [], []

    for submission in tqdm_notebook(subreddit.hot(limit=limit)):
        vid_id = submission.id
        if submission.is_video:
            if submission.media['reddit_video']['duration'] >= 15:
                meta_data = submission.media['reddit_video']
                meta_data['title'] = submission.title
                meta_data['id'] = vid_id
                meta_data['upvote'] = submission.score
                videos_metadata.append(meta_data)
                comments = []
                for comment in submission.comments:
                    comments.append({
                        'id': vid_id,
                        'comment_id': comment.id,
                        'comment_body': comment.body,
                        'replies': get_comment_replies(comment),
                        'upvote': comment.score,
                        'name': comment.name,
                        'parent_id': comment.parent_id,
                        'subreddit': comment.subreddit.display_name,
                        'subreddit_id': comment.subreddit_id
                    })
                videos_comment.extend(comments)
    return videos_metadata, videos_comment


def download_videos(metadata_df, download_path='vids'):
    """
    For a given dataframe with ``fallback_url`` and ``id`` as columns and path to download,
    download videos and save to a given ``download_path`` folder
    """
    if not os.path.isdir(download_path):
        os.mkdir(download_path)
    for _, r in tqdm_notebook(metadata_df.iterrows()):
        if r['duration'] <= 50:
            download_path = os.path.join(
                download_path, '{}.mp4'.format(r['id']))
            if not os.path.exists(download_path):
                try:
                    urllib.request.urlretrieve(
                        r['fallback_url'], download_path)
                except:
                    print(r['id'])


if __name__ == '__main__':
    videos_metadata, videos_comment = fetch_subreddit_info(limit=4000)
    download_videos(metadata_df, download_path='vids')
    # save data to JSON
    json.dump(videos_metadata, open('metadata.json', 'w'), indent=2)
    json.dump(videos_comment, open('comments.json', 'w'), indent=2)
