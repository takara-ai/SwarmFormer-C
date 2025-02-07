#ifndef PROFILER_H
#define PROFILER_H

#include <stdbool.h>

#define MAX_ENTRIES 100
#define MAX_NAME_LENGTH 50

typedef struct {
    char name[MAX_NAME_LENGTH];
    double start_time;
    double duration;
    double flops;
} ProfileEntry;

typedef struct {
    ProfileEntry entries[MAX_ENTRIES];
    int num_entries;
    bool enabled;
    int current_iter;
    int total_iters;
} Profiler;

extern Profiler profiler;

void print_progress_bar(int current, int total, int width);
void start_profile(const char* name);
void end_profile(const char* name, double flops);
void reset_profiler();
void start_benchmark(int num_iterations);
void end_benchmark();

#endif 