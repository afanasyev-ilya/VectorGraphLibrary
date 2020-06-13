#!/bin/bash
#SBATCH --partition aurora
#SBATCH --gres=ve:1
#SBATCH --exclusive

./$1 -load ./ext_csr_graphs/dir_rmat_20_32_ext_CSR.gbin $2
./$1 -load ./ext_csr_graphs/dir_rmat_21_32_ext_CSR.gbin $2
./$1 -load ./ext_csr_graphs/dir_rmat_22_32_ext_CSR.gbin $2
./$1 -load ./ext_csr_graphs/dir_rmat_23_32_ext_CSR.gbin $2
./$1 -load ./ext_csr_graphs/dir_rmat_24_32_ext_CSR.gbin $2
./$1 -load ./ext_csr_graphs/dir_rmat_25_32_ext_CSR.gbin $2
./$1 -load ./ext_csr_graphs/dir_rmat_26_32_ext_CSR.gbin $2
./$1 -load ./ext_csr_graphs/dir_rmat_27_32_ext_CSR.gbin $2

./$1 -load ./ext_csr_graphs/dir_ru_20_32_ext_CSR.gbin $2
./$1 -load ./ext_csr_graphs/dir_ru_21_32_ext_CSR.gbin $2
./$1 -load ./ext_csr_graphs/dir_ru_22_32_ext_CSR.gbin $2
./$1 -load ./ext_csr_graphs/dir_ru_23_32_ext_CSR.gbin $2
./$1 -load ./ext_csr_graphs/dir_ru_24_32_ext_CSR.gbin $2
./$1 -load ./ext_csr_graphs/dir_ru_25_32_ext_CSR.gbin $2

./$1 -load ./ext_csr_graphs/twitter_ext_CSR.gbin $2
./$1 -load ./ext_csr_graphs/orkut_ext_CSR.gbin $2
./$1 -load ./ext_csr_graphs/lj_ext_CSR.gbin $2
./$1 -load ./ext_csr_graphs/pokec_ext_CSR.gbin $2

./$1 -load ./ext_csr_graphs/wiki_en_ext_CSR.gbin $2
./$1 -load ./ext_csr_graphs/dbpedia_ext_CSR.gbin $2
./$1 -load ./ext_csr_graphs/trackers_ext_CSR.gbin $2
./$1 -load ./ext_csr_graphs/wiki_fr_ext_CSR.gbin $2
./$1 -load ./ext_csr_graphs/wiki_ru_ext_CSR.gbin $2