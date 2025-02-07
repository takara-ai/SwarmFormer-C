#ifndef PROFILER_H
#define PROFILER_H

#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

typedef struct {
    const char* name;
    double total_time;
    double total_flops;
    int calls;
} ProfileStat;

typedef struct {
    ProfileStat stats[32];
    int num_stats;
    bool enabled;
} Profiler;

extern Profiler profiler;

void init_profiler(void);
void enable_profiler(bool enable);
void start_profile(const char* name);
void end_profile(const char* name, double flops);
void print_profile_stats(void);

#endif 