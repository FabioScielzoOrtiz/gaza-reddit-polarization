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

        processed_data = processed_data.with_columns(
            pl.col("extraction_query").replace(query_map).alias("extraction_query")
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