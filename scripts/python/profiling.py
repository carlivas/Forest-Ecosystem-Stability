import pstats

# Load the profiling results
profile_file = './sim_run_profile.prof'
stats = pstats.Stats(profile_file)

# Sort and print the profiling results
stats.strip_dirs()  # Remove extraneous path info
stats.sort_stats('time')  # Sort by total time spent in the function
stats.print_stats(20)  # Print the top 20 functions
