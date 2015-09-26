#include "cuLife.cuh"
#include <time.h>
#include <algorithm>

__global__ void
Search(uint64_t * blck, LifeIterator *blckiter1, LifeIterator *blckiter2, LifeIterator *glditer, uint64_t * blcks1, uint64_t * blcks2, uint64_t * glds, unsigned int* lastIdx, uint64_t* results, const int maxResults)
{
	uint64_t state[N];
	uint64_t idx = ITER_PER_THREAD * (THREADS * blockIdx.x + threadIdx.x);

	for(uint64_t itIdx = idx; itIdx < idx + ITER_PER_THREAD; itIdx++)
	{
		int3 val[3];
		FindState(blckiter1, blckiter2, glditer, itIdx, val);

		if(val[2].z == -1)
			return;

		if(IsFirstBigger(val[0], val[1]) == NO)
			continue;

		ClearData(state);
		
		Append(state, blck);
		Append(state, blckiter1, blcks1, val[0]);
		Append(state, blckiter2, blcks2, val[1]);
		
		int stable = YES; 
		uint64_t hash = HashCode(state);
		
		for(int i = 0; i < 6; i++)
		{
			int pop = GetPop(state);
		
			if(pop != 4 * 3 || HashCode(state) != hash)
			{
				stable = NO;
				break;
			}
		
			IterateState(state, 0, N - 1);
		}

		if(stable == NO)
			continue;

		Append(state, glditer, glds, val[2]);
		
		for(int i = 0; i < 200; i++)
		{
			IterateState(state, 0, N - 1);
		}

		if(GetPop(state) == 5)
		{
			uint64_t hash = HashCode(state);
			IterateState(state, 0, N - 1);

			if(HashCode(state) != hash &&  GetPop(state) == 5)
			{
				int idx = atomicInc(lastIdx, maxResults);
				results[idx] = itIdx;
			}
		}
	}
}


int
main(void)
{

	const int maxResult = 50000;
	
	time_t	tic = clock();

	uint64_t * blck =  NewState("2o$2o!");
	uint64_t * gld =  NewState("2o$obo$o!");
	
	LifeIterator *blckiter1 = NewIterator(-10, -10, 20, 10, 1);
	LifeIterator *blckiter2 = NewIterator(-10, -10, 20, 10, 1);
	LifeIterator *glditer = NewIterator(-15, 5, 35, 1, 1);


	uint64_t * blcks1 = StatesContainer(blck, 1);
	uint64_t * blcks2 = StatesContainer(blck, 1);
	uint64_t * glds = StatesContainer(gld, 1);

	long numBlocks = NumBlocks(blckiter1, blckiter2, glditer);
	
	unsigned int* lastIdx = CudaNewPointer<unsigned int>() ;
	uint64_t * results = CudaNewArray<uint64_t>(maxResult);
	
	LifeIterator *d_blckiter1 = CopyToDevice(blckiter1);
	LifeIterator *d_blckiter2 = CopyToDevice(blckiter2);
	LifeIterator *d_glditer = CopyToDevice(glditer);

	uint64_t * d_blcks1 = CudaCopyArrayToDevice(blcks1, 1 * N);
	uint64_t * d_blcks2 = CudaCopyArrayToDevice(blcks2, 1 * N);
	uint64_t * d_glds = CudaCopyArrayToDevice(glds, 1 * N);

	uint64_t * d_blck = CudaCopyArrayToDevice(blck, N);
	
	Search<<<numBlocks, THREADS>>>(d_blck, d_blckiter1, d_blckiter2, d_glditer, d_blcks1, d_blcks2, d_glds, lastIdx, results, maxResult);
	
	cudaDeviceSynchronize();
	
	time_t	 toc = clock();
	printf("Elapsed: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
	
	unsigned int* h_lastIdx = CudaCopyPointerToHost(lastIdx);
	uint64_t * h_results = CudaCopyArrayToHost(results, *h_lastIdx);
	
	uint64_t state[N];
	
	printf("Total Results = %llu\n", (*h_lastIdx));

	for(int i = 0; i < (int)(*h_lastIdx); i++)
	{
		ClearData(state);
		
		int3 val[3];
		FindState(blckiter1, blckiter2, glditer,  h_results[i], val);
		
		Append(state, blck);
		Append(state, blckiter1, blcks1, val[0]);
		Append(state, blckiter2, blcks2, val[1]);
		Append(state, glditer, glds, val[2]);
		
		printf("\nResult %d of %d\n", i + 1, *h_lastIdx);
		
		//Print(state);
		PrintRLE(state);
		
		//getchar();
		/*
		for(int j = 0; j < 200; j++)
		{
			IterateState(state, 0, N - 1);
			Print(state);
			if(getchar() == 'q')
				break;
		}
		*/
	}
	
	cudaDeviceReset();
	
	getchar();
	
    return 0;
}
