#include "gonio.h"


const int16_t sine_table[SINE_TABLE_SIZE] = {
512,515,518,521,525,528,531,534,
537,540,543,547,550,553,556,559,
562,565,568,572,575,578,581,584,
587,590,593,596,600,603,606,609,
612,615,618,621,624,627,630,633,
636,639,642,646,649,652,655,658,
661,664,667,670,673,676,679,682,
684,687,690,693,696,699,702,705,
708,711,714,717,719,722,725,728,
731,734,737,739,742,745,748,751,
753,756,759,762,764,767,770,773,
775,778,781,783,786,789,791,794,
796,799,802,804,807,809,812,814,
817,820,822,825,827,829,832,834,
837,839,842,844,846,849,851,854,
856,858,860,863,865,867,870,872,
874,876,878,881,883,885,887,889,
891,893,896,898,900,902,904,906,
908,910,912,914,916,918,919,921,
923,925,927,929,931,932,934,936,
938,939,941,943,945,946,948,950,
951,953,954,956,957,959,961,962,
964,965,966,968,969,971,972,973,
975,976,977,979,980,981,983,984,
985,986,987,989,990,991,992,993,
994,995,996,997,998,999,1000,1001,
1002,1003,1004,1005,1005,1006,1007,1008,
1009,1009,1010,1011,1012,1012,1013,1014,
1014,1015,1015,1016,1016,1017,1017,1018,
1018,1019,1019,1020,1020,1021,1021,1021,
1022,1022,1022,1022,1023,1023,1023,1023,
1023,1024,1024,1024,1024,1024,1024,1024,
1024,1024,1024,1024,1024,1024,1024,1024,
1023,1023,1023,1023,1023,1022,1022,1022,
1022,1021,1021,1021,1020,1020,1019,1019,
1018,1018,1017,1017,1016,1016,1015,1015,
1014,1014,1013,1012,1012,1011,1010,1009,
1009,1008,1007,1006,1005,1005,1004,1003,
1002,1001,1000,999,998,997,996,995,
994,993,992,991,990,989,987,986,
985,984,983,981,980,979,977,976,
975,973,972,971,969,968,966,965,
964,962,961,959,957,956,954,953,
951,950,948,946,945,943,941,939,
938,936,934,932,931,929,927,925,
923,921,919,918,916,914,912,910,
908,906,904,902,900,898,896,893,
891,889,887,885,883,881,878,876,
874,872,870,867,865,863,860,858,
856,854,851,849,846,844,842,839,
837,834,832,829,827,825,822,820,
817,814,812,809,807,804,802,799,
796,794,791,789,786,783,781,778,
775,773,770,767,764,762,759,756,
753,751,748,745,742,739,737,734,
731,728,725,722,719,717,714,711,
708,705,702,699,696,693,690,687,
684,682,679,676,673,670,667,664,
661,658,655,652,649,646,642,639,
636,633,630,627,624,621,618,615,
612,609,606,603,600,596,593,590,
587,584,581,578,575,572,568,565,
562,559,556,553,550,547,543,540,
537,534,531,528,525,521,518,515,
512,509,506,503,499,496,493,490,
487,484,481,477,474,471,468,465,
462,459,456,452,449,446,443,440,
437,434,431,428,424,421,418,415,
412,409,406,403,400,397,394,391,
388,385,382,378,375,372,369,366,
363,360,357,354,351,348,345,342,
340,337,334,331,328,325,322,319,
316,313,310,307,305,302,299,296,
293,290,287,285,282,279,276,273,
271,268,265,262,260,257,254,251,
249,246,243,241,238,235,233,230,
228,225,222,220,217,215,212,210,
207,204,202,199,197,195,192,190,
187,185,182,180,178,175,173,170,
168,166,164,161,159,157,154,152,
150,148,146,143,141,139,137,135,
133,131,128,126,124,122,120,118,
116,114,112,110,108,106,105,103,
101,99,97,95,93,92,90,88,
86,85,83,81,79,78,76,74,
73,71,70,68,67,65,63,62,
60,59,58,56,55,53,52,51,
49,48,47,45,44,43,41,40,
39,38,37,35,34,33,32,31,
30,29,28,27,26,25,24,23,
22,21,20,19,19,18,17,16,
15,15,14,13,12,12,11,10,
10,9,9,8,8,7,7,6,
6,5,5,4,4,3,3,3,
2,2,2,2,1,1,1,1,
1,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,
1,1,1,1,1,2,2,2,
2,3,3,3,4,4,5,5,
6,6,7,7,8,8,9,9,
10,10,11,12,12,13,14,15,
15,16,17,18,19,19,20,21,
22,23,24,25,26,27,28,29,
30,31,32,33,34,35,37,38,
39,40,41,43,44,45,47,48,
49,51,52,53,55,56,58,59,
60,62,63,65,67,68,70,71,
73,74,76,78,79,81,83,85,
86,88,90,92,93,95,97,99,
101,103,105,106,108,110,112,114,
116,118,120,122,124,126,128,131,
133,135,137,139,141,143,146,148,
150,152,154,157,159,161,164,166,
168,170,173,175,178,180,182,185,
187,190,192,195,197,199,202,204,
207,210,212,215,217,220,222,225,
228,230,233,235,238,241,243,246,
249,251,254,257,260,262,265,268,
271,273,276,279,282,285,287,290,
293,296,299,302,305,307,310,313,
316,319,322,325,328,331,334,337,
340,342,345,348,351,354,357,360,
363,366,369,372,375,378,382,385,
388,391,394,397,400,403,406,409,
412,415,418,421,424,428,431,434,
437,440,443,446,449,452,456,459,
462,465,468,471,474,477,481,484,
487,490,493,496,499,503,506,509
};


int16_t sin_tab(uint16_t angle)
{
    unsigned int idx = (angle%SINE_TABLE_SIZE);
    return sine_table[idx] - SINE_TABLE_MAX;
}

int16_t cos_tab(uint16_t angle)
{
    return sin_tab(angle + SINE_TABLE_SIZE/4);
}
