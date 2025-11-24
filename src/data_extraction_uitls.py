# data_extraction_uitls.py

import praw
import prawcore
import time
import datetime as dt
import logging
import itertools

# --- AUTHENTICATION ---

def authenticate_praw(CLIENT_ID, CLIENT_SECRET, USER_AGENT):
    """Initializes the PRAW instance in read-only mode."""
    try:
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT,
        )
        reddit.user.me() 
        logging.info("PRAW instance initialized successfully (read-only mode).")
        return reddit
    except Exception as e:
        logging.error(f"Authentication failed: {e}")
        return None

# --- HELPER FUNCTION: AUTHOR DATA EXTRACTION ---

'''
def _get_author_data(author_obj):
    """
    Safely extracts author data (ID, name, karma, age) from a PRAW Author object.
    Handles deleted/suspended accounts by returning default values.
    """
    # Default values for deleted/suspended users
    default_data = {
        'author_id': '[unavailable]',
        'author_name': '[deleted]',
        'author_post_karma': 0,
        'author_comment_karma': 0,
        'account_age_days': -1,
        'author_status': 'Deleted'
    }
    
    if author_obj is None:
        return default_data
        
    try:
        # Check for suspended accounts
        if author_obj.is_suspended:
            default_data['author_name'] = '[suspended]'
            default_data['author_status'] = 'Suspended'
            return default_data

        # Calculate account age in days
        created_utc = author_obj.created_utc
        account_age_days = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromtimestamp(created_utc, dt.timezone.utc)).days

        return {
            'author_id': author_obj.id,
            'author_name': author_obj.name,
            'author_post_karma': author_obj.link_karma,
            'author_comment_karma': author_obj.comment_karma,
            'account_age_days': account_age_days,
            'author_status': 'Active'
        }
    except Exception:
        # Fallback for unexpected errors or other inaccessible accounts
        default_data['author_status'] = 'Error'
        return default_data
'''
        
# --- CORE EXTRACTION FUNCTION ---

def run_extraction(reddit, subreddits, queries, sorts, max_limit, time_filter):
    """
    Runs the full extraction process over all combinations of subreddits, queries, and sorts.
    Always extracts posts AND top-level comments for the Comment-Centric strategy.
    """
    post_data_list = []
    comment_data_list = []
    post_ids_seen = set()
    comment_ids_seen = set()
    extraction_time_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    
    # Generate all combinations of (subreddit, query, sort)
    extraction_combinations = list(itertools.product(subreddits, queries, sorts))
    logging.info(f"Total extraction combinations to run: {len(extraction_combinations)}")

    for subreddit_name, query, sort in extraction_combinations:
        
        logging.info(f"Processing r/{subreddit_name} | Query: '{query[:30]}...' | Sort: {sort}")

        try:
            # 1. Search for posts
            search_results = reddit.subreddit(subreddit_name).search(
                query,
                sort=sort,
                time_filter=time_filter,
                limit=max_limit
            )
            
            # 2. Iterate over posts
            for post in search_results:
                if post.id in post_ids_seen:
                    continue # Skip if already processed
                logging.info(f'⚙️ Extracting data from post {post.id}')

                # 2a. Extract Post Data
                #author_data = _get_author_data(post.author)
                
                post_record = {
                    # Identifiers
                    'post_id': post.id,
                    'post_subreddit': post.subreddit.display_name,
                    # Text Content
                    'post_title': post.title,
                    'post_body': post.selftext,
                    'post_url': post.url,
                    # Post Metrics
                    'post_score': post.score,
                    'post_upvote_ratio': post.upvote_ratio,
                    'post_num_comments': post.num_comments,
                    'post_num_crossposts': post.num_crossposts,
                    'post_total_awards': post.total_awards_received,
                    'post_is_self': post.is_self,
                    'post_is_over_18': post.over_18,
                    'post_is_stickied': post.stickied,
                    'post_is_locked': post.locked,
                    'post_subreddit_subscribers': post.subreddit_subscribers,
                    'post_domain': post.domain,
                    'post_flair': post.link_flair_text if post.link_flair_text else None,
                    # Author Data
                    #'author_id': author_data['author_id'],
                    #'author_name': author_data['author_name'],
                    #'author_post_karma': author_data['author_post_karma'],
                    #'author_comment_karma': author_data['author_comment_karma'],
                    #'account_age_days': author_data['account_age_days'],
                    #'author_status': author_data['author_status'],
                    # Timestamps
                    'post_created_utc': post.created_utc,
                    'post_created_utc_date': dt.datetime.fromtimestamp(post.created_utc, dt.timezone.utc).isoformat(),
                    # Traceability Metadata
                    'extraction_query': query,
                    'extraction_sort': sort,
                    'extraction_time': extraction_time_utc,
                }
                
                post_data_list.append(post_record)
                post_ids_seen.add(post.id)
                logging.info(f'✅ Post {post.id}')

                # 2b. Extract Top-Level Comments (New universal logic)
                try:
                    # Replace 'MoreComments' links to fetch top-level comments only (limit=0)
                    post.comments.replace_more(limit=0)
                    
                    logging.info(f'⚙️ Extracting data from comments of post {post.id}')
                    for comment in post.comments.list():
                        if comment.id in comment_ids_seen:
                            continue # Skip if already processed

                        #author_data_comment = _get_author_data(comment.author)
                        
                        comment_record = {
                            # Identifiers
                            'comment_id': comment.id,
                            'post_id': post.id,
                            # Text Content
                            'comment_body': comment.body,
                            # Comment Metrics
                            'comment_score': comment.score,
                            'comment_score_hidden': comment.score_hidden,
                            # Author Data
                            #'author_id': author_data_comment['author_id'],
                            #'author_name': author_data_comment['author_name'],
                            #'author_post_karma': author_data_comment['author_post_karma'],
                            #'author_comment_karma': author_data_comment['author_comment_karma'],
                            #'account_age_days': author_data_comment['account_age_days'],
                            #'author_status': author_data_comment['author_status'],
                            # Timestamps
                            'comment_created_utc': comment.created_utc,
                            'comment_created_utc_date': dt.datetime.fromtimestamp(comment.created_utc, dt.timezone.utc).isoformat(),
                        }
                        
                        comment_data_list.append(comment_record)
                        comment_ids_seen.add(comment.id)
                        logging.info(f'✅ Comment {comment.id}')

                except Exception as e:
                    logging.warning(f"❌ Error retrieving comments for post {post.id} in r/{subreddit_name}: {e}")
            
            logging.info(f"-> Unique POSTS: {len(post_ids_seen)} | Unique COMMENTS: {len(comment_ids_seen)}")
            
        except prawcore.exceptions.NotFound:
            logging.error(f"❌ -> ERROR! Subreddit r/{subreddit_name} not found (404). Skipping.")
        except prawcore.exceptions.Forbidden:
            logging.error(f"❌ -> ERROR! Subreddit r/{subreddit_name} is private or banned (403). Skipping.")
        except Exception as e:
            logging.error(f"❌ -> UNEXPECTED ERROR in r/{subreddit_name}: {e}. Skipping.")
        
        # Pause to respect API rate limits
        time.sleep(1.2)
        
    logging.info(f"Data extraction completed. Total unique posts: {len(post_ids_seen)}. Total unique comments: {len(comment_ids_seen)}.")
    return post_data_list, comment_data_list