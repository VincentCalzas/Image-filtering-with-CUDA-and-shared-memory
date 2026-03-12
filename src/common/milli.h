#ifndef _MILLI_
#define _MILLI_

#ifdef __cplusplus
extern "C" {
#endif

int GetMilliseconds();
int GetMicroseconds();
double GetSeconds();

// Optional setting of the start time. If these are not used,
// the first call to the above functions will be the start time.
void ResetMilli();
void SetMilli(int seconds, int microseconds);

#ifdef __cplusplus
}
#endif

#endif
