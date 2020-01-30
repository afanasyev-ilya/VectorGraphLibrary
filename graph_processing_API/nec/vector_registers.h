/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define NEC_REGISTER_INT(name, value)\
int reg_##name[VECTOR_LENGTH];\
for(int i = 0; i < VECTOR_LENGTH; i++)\
{\
    reg_##name[i] = 0;\
}

#define NEC_REGISTER_FLT(name, value)\
float reg_##name[VECTOR_LENGTH];\
for(int i = 0; i < VECTOR_LENGTH; i++)\
{\
    reg_##name[i] = 0;\
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
