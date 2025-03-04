#!/usr/bin/env python
"""
Script to fix issues in the experiment log CSV.
- Identifies and corrects missing or invalid dataset names
- Fills in missing values where possible (regression slopes, time metrics)
- Handles existing issues in the log while future runs will have proper error handling
"""

from src.managers.experiment_tracker import ExperimentTracker
from src.managers.logging_manager import LoggingManager

def main():
    logger = LoggingManager()
    logger.section("Experiment Log Fix Utility")
    
    # Create tracker instance
    tracker = ExperimentTracker()
    
    # Run the fix method
    logger.info("Starting experiment log fixes...")
    success = tracker.fix_experiment_log()
    
    if success:
        logger.success("Successfully fixed experiment log issues")
        
        # Show stats about the fixed data
        df = tracker.df
        logger.info(f"Total experiments: {len(df)}")
        
        # Show unique datasets
        datasets = df['dataset'].unique().tolist()
        logger.info(f"Unique datasets: {datasets}")
        
        # Show warning if 'unknown' is in datasets
        if 'unknown' in datasets or 'UNKNOWN' in datasets:
            logger.warning("Some experiments have 'unknown' dataset values. These should be examined.")
            logger.info("Future experiment runs will raise errors for undefined datasets instead of using defaults.")
        
        # Count completed experiments
        completed = df[df['status'] == 'complete']
        logger.info(f"Completed experiments: {len(completed)}")
        
        # Show field completion rates for previously problematic fields
        for field in ['regression_slope', 'time_per_epoch', 'best_epoch', 'layer_irreps', 'max_epochs', 'num_features', 'n_val']:
            non_null = df[field].notnull().sum()
            logger.info(f"Field '{field}' completion rate: {non_null}/{len(df)} ({non_null/len(df)*100:.1f}%)")
    else:
        logger.error("Failed to fix experiment log")
        
if __name__ == "__main__":
    main() 