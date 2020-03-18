#include "ap_fixed.h"
#include <parameters.h>

#define STREAMSIZE 16384
//how many consecutive sets of inputs to run over per kernel execution

#define DATA_SIZE_IN N_INPUT_1_1
#define DATA_SIZE_OUT N_LAYER_6
#define COMPRESSION 16
#define COMPSTREAMSIZE 1024

typedef ap_fixed<32,14> data_t;
typedef ap_uint<512>    bigdata_t;
