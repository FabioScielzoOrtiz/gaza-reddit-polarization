#################################################################################################

# --- IMPORTS ---

import os, sys
import logging
import polars as pl

#################################################################################################

# --- LOGGING CONFIGURATION ---

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#################################################################################################

# --- PATH CONFIGURATION ---

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..')
sys.path.append(project_path)

# Input / Output:
processed_data_dir = os.path.join(project_path, 'data', 'processed_data')
processed_data_path = os.path.join(processed_data_dir, '05c_processed_data.parquet')
output_path = os.path.join(processed_data_dir, '06_processed_data.parquet')

#################################################################################################

# --- MAIN EXECUTION ---

def main():

    # 1. Load Data
    try:
        processed_data = pl.read_parquet(processed_data_path)
        logging.info(f"Base dataset loaded: {len(processed_data)} records.")
    except Exception as e:
        logging.error(f"❌ Failed to load processed data: {e}")
        exit()

    # 2. Process Data
    try:
        # Datetime casting
        processed_data = processed_data.with_columns([
            pl.col(c).str.to_datetime(format="%Y-%m-%dT%H:%M:%S%z", strict=False)
            for c in ["post_created_date", "comment_created_date"]
        ]).with_columns(
            pl.col("extraction_time").str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.f%z", strict=False)
        )

        # Query mapping
        query_map = {
            '(gaza OR palestine*)': 'General',
            '((gaza OR palestine*) AND (genocide OR humanitarian* OR UN OR ICJ OR victims OR war crime)) OR (israel AND (genocide OR humanitarian* OR UN OR ICJ OR victims OR war crime))': 'Humanitarian/Legal',
            '((gaza OR palestine*) AND (hamas OR terroris* OR attack OR hostages OR netanyahu OR IDF OR war OR conflict)) OR (israel AND (hamas OR terroris* OR attack OR hostages OR netanyahu OR IDF OR war OR conflict))': 'Conflict/Security',
            '((gaza OR palestine*) AND (biden OR trump OR US OR congress OR EU OR ally OR antisemit* OR "anti-semit*")) OR (israel AND (biden OR trump OR US OR congress OR EU OR ally OR antisemit* OR "anti-semit*"))': 'Geopolitical/Political',
            '((gaza OR palestine*) AND (media OR narrative OR propaganda OR bias OR reporting OR antisemit* OR "anti-semit*" OR islamophob* OR misinformation OR "hate speech")) OR (israel AND (media OR narrative OR propaganda OR bias OR reporting OR antisemit* OR "anti-semit*" OR islamophob* OR misinformation OR "hate speech"))': 'Media/Narrative',
        }

        political_stance_map = {
            "1": 'Strongly Pro-Palestine',
            "2": 'Leaning Pro-Palestine',
            "3": 'Neutral / Balanced',
            "4": 'Leaning Pro-Israel',
            "5": 'Strongly Pro-Israel',
            "-1": 'Unclassifiable'
        }

        discourse_tone_map = {
            "1": 'Analytical',
            "2": 'Emotional',
            "3": 'Hostile',
            "4": 'Sarcastic',
            "5": 'Informative',
            "6": 'Other'
        }

        dominant_frame_map = {
            "1": 'Humanitarian',
            "2": 'Legal',
            "3": 'Security/Military',
            "4": 'Historical/Identity',
            "5": 'Media/Narrative',
            "6": 'Geopolitical',
            "7": 'Ideological',
            "8": 'Other'
        }

        argument_quality_map = {
            "0": 'Spam / Non-Argument',
            "1": 'Pure Reaction',
            "2": 'Bare Opinion',
            "3": 'Justified Opinion',
            "4": 'Reasoned Argument',
            "5": 'Sophisticated Discourse'
        }

        # Sentiment continuous logic using conditional expressions
        sentiment_condition = (
            pl.when(pl.col("sentiment_score") >= 0.7).then(pl.lit("Very Positive"))
            .when(pl.col("sentiment_score") >= 0.4).then(pl.lit("Moderately Positive"))
            .when(pl.col("sentiment_score") >= 0.1).then(pl.lit("Mildly Positive"))
            .when(pl.col("sentiment_score") == 0.0).then(pl.lit("Neutral"))
            .when(pl.col("sentiment_score") >= -0.3).then(pl.lit("Mildly Negative"))
            .when(pl.col("sentiment_score") >= -0.6).then(pl.lit("Moderately Negative"))
            .otherwise(pl.lit("Very Negative"))
        )

        # Apply mappings (Casteando a pl.String antes de reemplazar)
        processed_data = processed_data.with_columns(
            pl.col("extraction_query").replace(query_map).alias("extraction_query"),
            pl.col("political_stance_score").cast(pl.String).replace(political_stance_map).alias("political_stance_score_label"),
            pl.col("discourse_tone_score").cast(pl.String).replace(discourse_tone_map).alias("discourse_tone_score_label"),
            pl.col("dominant_frame_score").cast(pl.String).replace(dominant_frame_map).alias("dominant_frame_score_label"),
            pl.col("argument_quality_score").cast(pl.String).replace(argument_quality_map).alias("argument_quality_score_label"),
            sentiment_condition.alias("sentiment_score_label")
        )

        # Filtering invalid scores
        processed_data = processed_data.filter(
            pl.col('political_stance_score') != -1,
            pl.col('discourse_tone_score') != -1,
            ~ pl.col('dominant_frame_score').is_in([-1, 9]),
            pl.col('argument_quality_score') != -1
        )

        logging.info(f"✅ Data processing applied successfully. Remaining records: {len(processed_data)}.")

    except Exception as e:
        logging.error(f"❌ Error during data processing: {e}")
        exit()

    # 3. Save Data
    try:
        processed_data.write_parquet(output_path)
        logging.info(f'📁 Processed file saved at {output_path}.')
    except Exception as e:
        logging.error(f"❌ Failed to save processed data: {e}")
        exit()

if __name__ == "__main__":
    main()

#################################################################################################