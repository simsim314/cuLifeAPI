#include <stdio.h>

typedef unsigned long long uint64_t;
#define N 64
#define cuDef __device__ __host__ 
#define SUCCESS 1
#define FAIL 0

#define YES 1
#define NO 0
#define MAX_ITERATIONS 200

#define ITER_PER_THREAD 10
#define THREADS 256


enum CopyType { COPY, OR, XOR, AND };
enum EvolveType { EVOLVE, LEAVE };

template<class T>
cuDef T* CudaNewPointer()
{
    T* result; 
	cudaMalloc((void**)&result, sizeof(T));
	cudaMemset(result, 0, sizeof(T));
	return result;
}

template<class T>
cuDef T* CudaNewArray(int size)
{
    T* result; 
	cudaMalloc((void**)&result, size * sizeof(T));
	cudaMemset(result, 0, size * sizeof(T));
	return result; 
}

template<class T>
cuDef T* HostNewPointer()
{
    T* result = (T*)malloc(sizeof(T));
	memset(result, 0, sizeof(T));
	return result;
}

template<class T>
cuDef T* HostNewArray(int size)
{
    T* result = (T*)malloc(size * sizeof(T));
	memset(result, 0, size * sizeof(T));
	return result; 
}


template<class T>
cuDef T* CudaCopyPointerToHost(T* cuPnt)
{
    T* result; 
	result = (T*)malloc(sizeof(T));
	cudaMemcpy(result, cuPnt, sizeof(T), cudaMemcpyDeviceToHost);
	return result;
}

template<class T>
cuDef T* CudaCopyArrayToHost(T* cuPnt, int size)
{
    T* result; 
	result = (T*)malloc(size * sizeof(T));
	cudaMemcpy(result, cuPnt, size * sizeof(T), cudaMemcpyDeviceToHost);
	return result;
}

template<class T>
cuDef T* CudaCopyPointerToDevice(T* p)
{
    T* result = CudaNewPointer<T>(); 
	cudaMemcpy(result, p, sizeof(T), cudaMemcpyHostToDevice);
	return result;
}

template<class T>
cuDef void CudaCopyPointerToDevice(T* cuPnt, T* p)
{
	cudaMalloc((void**)&cuPnt, sizeof(T));
	cudaMemcpy(cuPnt, p, sizeof(T), cudaMemcpyHostToDevice);
	return result;
}

template<class T>
cuDef T* CudaCopyArrayToDevice(T* p, int size)
{
    T* result = CudaNewArray<T>(size); 
	cudaMemcpy(result, p, size * sizeof(T), cudaMemcpyHostToDevice);
	return result;
}

template<class T>
cuDef void CudaCopyArrayToDevice(T* target, T* p, int size)
{
	cudaMalloc((void**)&target, size * sizeof(T));
	cudaMemcpy(target, p, size * sizeof(T), cudaMemcpyHostToDevice);
}

typedef struct 
{
	char* value;
	int size;
	int allocated;
	
} LifeString;

LifeString* NewString()
{
	LifeString* result = (LifeString*)(malloc(sizeof(LifeString)));
	
	result->value = (char*)(malloc(2 * sizeof(char)));
	result->value[0] = '\0';
	result->size = 1;
	result->allocated = 1;
	
	return result;
}

void Realloc(LifeString* string)
{
	int empty = NO;
	
	if(string->value[0] == '\0')
		empty = YES;
	
	if(empty == NO)
	{
		string->value = (char*)(realloc(string->value, string->allocated * 2 * sizeof(char)));
	}
	else
	{
		string->value = (char*)(malloc(string->allocated * 2 * sizeof(char)));
		string->value[0] = '\0';
	}
	
	string->allocated *= 2;
}

void Realloc(LifeString* string, int size)
{
	while(string->allocated <= string->size + size + 1)
		Realloc(string);
}

void Append(LifeString* string, const char* val)
{
	Realloc(string, strlen(val));
	strcat(string->value, val);
	string->size = strlen(string->value);
}

void Append(LifeString* string, int val)
{
	char str[10];
	sprintf(str, "%d", val);
	Append(string, str);
}

LifeString* NewString(const char* val)
{
	LifeString* result = NewString();
	Append(result, val);
	return result;
}


cuDef __forceinline__ uint64_t CirculateLeft(uint64_t x)
{
	return (x << 1) | (x >> (63));
}

cuDef __forceinline__ uint64_t CirculateRight(uint64_t x)
{
	return (x >> 1) | (x << (63));
}

cuDef  uint64_t CirculateLeft(uint64_t x, int k)
{
	if(k == 0 || k == 64)
		return x;

	return (x << k) | (x >> (64 - k));
}

cuDef uint64_t CirculateRight(uint64_t x, int k)
{
	if(k == 0 || k == 64)
		return x;

	return (x >> k) | (x << (64 - k));
}

cuDef int Get(int x, int y, uint64_t state[N])
{
	return (state[x] & (1ULL << y)) >> y;
}

cuDef int GetCell(uint64_t state[N], int x, int y)
{
	return Get((x + 32) % 64, (y + 32) % 64, state);
}

cuDef void Set(int x, int y, uint64_t  *state)
{
	state[x] |= (1ULL << (y));
}

cuDef void Erase(int x, int y, uint64_t  *state)
{
	state[x] &= ~(1ULL << (y));
}

cuDef void SetCell(uint64_t state[N], int x, int y, int val)
{
	if(val == 1)
		Set((x + 32) % N, (y + 32) % 64, state);
	else if(val == 0)
		Erase((x + 32) % 64, (y + 32) % 64,  state);
}

const char* GetRLE(uint64_t state[N])
{
    LifeString* result = NewString();
	
	int eol_count = 0; 

	for(int j = 0; j < 64; j++)
	{
		int last_val = -1;
		int run_count = 0;
		
        for(int i = 0; i < N; i++)
		{
			int val = Get(i, j, state);

			// Flush linefeeds if we find a live cell
			if(val == 1 && eol_count > 0)
			{
				if(eol_count > 1)
					Append(result, eol_count);
				
				Append(result, "$");
				
				eol_count = 0;
			}

			// Flush current run if val changes
			if (val == 1 - last_val)
			{
				if(run_count > 1)
					Append(result, run_count);
				
				Append(result, last_val ? "o" : "b");
				
				run_count = 0;
			}

			run_count++;
			last_val = val;
		}

		// Flush run of live cells at end of line
		if (last_val == 1)
		{
			if(run_count > 1)
				Append(result, run_count);
					
			Append(result, "o");
					
			run_count = 0;
		}

		eol_count++;
	}
        
	return result->value;
}

void PrintRLE(uint64_t *state)
{
    printf("\nx = 0, y = 0, rule = B3/S23\n%s!\n\n", GetRLE(state));
}

cuDef void Print(uint64_t state[N])
{


	int i, j;

	for(i = 0; i < N; i++)
	{
		for(j = 0; j < 64; j++)
		{
			if(GetCell(state, j - 32, i - 32) == 0)
			{
				int hor = 0;
				int ver = 0;
						
				if((i - 32) % 10 == 0)
					hor = 1;
						
				if((j - 32) % 10 == 0)
					ver = 1;
										
				if(hor == 1 && ver == 1)
					printf ("+");
				else if(hor == 1)
					printf ("-");
				else if(ver == 1)
					printf ("|");
				else
					printf (".");
			}
			else
				printf ("O");
		}
		printf("\n");
	}
			
	printf("\n\n\n\n\n\n");
	
}

cuDef void IterateState(uint64_t state[N], int min, int max)
{
	uint64_t tempxor[3];
	uint64_t tempand[3];

	uint64_t tempState;

	uint64_t l, r, tempVal, temp[3];
	uint64_t x0, r0, xU, aU, xB, aB;
	uint64_t a0,a1,a2,c, b0, b1, b2;

	int i, j, idxU, idxB;
	uint64_t  col0; 
	
	col0 = state[0];
		
	for(i = min; i <= max; i++)
	{
		
		if(i == min)
		{
			if(i == 0)
				idxU = N - 1;
			else
				idxU = i - 1;

			if(i == N - 1)
				idxB = 0;
			else
				idxB = i + 1;
				
			temp[0] = state[idxU];	
			temp[1] = state[i];	
			temp[2] = state[idxB];	
			
			#pragma unroll 
			for(j = 0; j <= 2; j++)
			{
				l = CirculateLeft(temp[j]);
				r = CirculateRight(temp[j]);
				tempxor[j] = l ^ r ^ temp[j];
				tempand[j] = ((l | r) & temp[j]) | (l & r);
			}
		}
		else
		{
			temp[1] = state[i];
			
			#pragma unroll 
			for(j = 0; j <= 1; j++)
			{
				tempxor[j] = tempxor[j + 1];
				tempand[j] = tempand[j + 1];
			}
			
			if(i == N - 1)
				tempVal = col0;
			else
				tempVal = state[i + 1];
				
			l = CirculateLeft(tempVal);
			r = CirculateRight(tempVal);
			tempxor[2] = l ^ r ^ tempVal;
			tempand[2] = ((l | r) & tempVal) | (l & r);
		}
		
		x0 = tempxor[1];
		r0 = tempand[1];

		xU = tempxor[0];
		aU = tempand[0];

		xB = tempxor[2];
		aB = tempand[2];

		a0 = x0^xU;
		c = (x0&xU);
		a1 = c^aU^r0;
		a2 = (aU&r0)|((aU|r0)&c);

		b0 = xB^a0;
		c = (xB&a0);
		b1 = c^aB^a1;
		b2 = (aB&a1)|((aB|a1)&c);
		
		if(i > min)
			state[i - 1] = tempState;
			
		tempState = (b0&b1&(~b2)&(~a2))|((temp[1])&(a2^b2)&(~b0)&(~b1));
	}
	
	state[max] = tempState;
}

cuDef void Reverse(uint64_t  *state, int idxS, int idxE)
{
	for(int i = 0; idxS + i <  idxE - i; i++)
	{
		int l = idxS + i; 
		int r = idxE - i;
		
		uint64_t temp = state[l];
		state[l] = state[r];
		state[r] = temp;
	}
}

cuDef void CirculateUp(uint64_t  *state, int k)
{
	Reverse(state, 0, N - 1);
	Reverse(state, 0, k - 1);
	Reverse(state, k, N - 1);
}

cuDef void Move(uint64_t* state, int x, int y)
{
	for(int i = 0; i < N; i++)
	{
		if(y < 0)
			state[i] = CirculateRight(state[i], -y);
		else
			state[i] = CirculateRight(state[i], 64 - y);
	}

	if(x < 0)
		CirculateUp(state, 64 + x);
	else
		CirculateUp(state, x);

}


cuDef void Parse(uint64_t* state, const char* rle)
{
	char ch;
	int cnt, i, j; 
	int x, y;
	x = 0;
	y = 0;
	cnt = 0;
	i = 0;
	
	while((ch = rle[i]) != '\0')
	{

		if(ch >= '0' && ch <= '9')
		{
			cnt *= 10;
			cnt += (ch - '0');
		}
		else if(ch == 'o')
		{
			
			if(cnt == 0)
				cnt = 1;
				
			for(j = 0; j < cnt; j++)
			{
				SetCell(state, x, y, 1);
				x++;
			}
			
			cnt = 0;
		}
		else if(ch == 'b')
		{
			if(cnt == 0)
				cnt = 1;
			
			x += cnt;
			cnt = 0;
			
		}
		else if(ch == '$')
		{
			if(cnt == 0)
				cnt = 1;
	
			y += cnt;
			x = 0;
			cnt = 0;
		}
		else if(ch == '!')
		{
			break;
		}
		else
		{
			return;
		}
		
		i++;
	}
}

cuDef void Parse(uint64_t state[N], const char* rle, int dx, int dy)
{
	Parse(state, rle);
	Move(state, dx, dy);
}


cuDef void Copy(uint64_t * main, uint64_t *  delta, CopyType op)
{
	if(op == COPY)
	{	
		for(int i = 0; i < N; i++)
			main[i] = delta[i];
	}
	else if(op == OR)
	{	
		for(int i = 0; i < N; i++)
			main[i] |= delta[i];
	}
	else if(op == AND)
	{	
		for(int i = 0; i < N; i++)
			main[i] &= delta[i];
		
	}
	else if(op == XOR)
	{	
		for(int i = 0; i < N; i++)
			main[i] ^= delta[i];
	}
}

cuDef void Copy(uint64_t *main, uint64_t *delta)
{
	Copy(main, delta, COPY);
}

__device__ int GetPop(uint64_t* state, int min, int max)
{
	int pop = 0;
	
	for(int i = min; i <= max; i++)
	{
		pop += __popcll(state[i]);
	}
	
	return pop;
}


__device__ int GetPop(uint64_t* state)
{
	return GetPop(state, 0, N - 1);
}

cuDef void Inverse(uint64_t* state)
{	
	for(int i = 0; i < N; i++)
	{
		state[i] = ~(state[i]);
	}
}

cuDef void ClearData(uint64_t* state)
{
	for(int i = 0; i < N; i++)
		state[i] = 0;
}

cuDef int AreEqual(uint64_t* pat1, uint64_t* pat2)
{
	for(int i = 0; i < N; i++)
		if(pat1[i] != pat2[i])
			return NO;
			
	return YES;
}


cuDef int AreDisjoint(uint64_t* mainState, uint64_t* patState, int min, int max)
{
	for(int i = min; i <= max; i++)
		if(((~mainState[i]) & patState[i]) != patState[i])
			return NO;
			
	return YES;
}

cuDef int Contains(uint64_t* mainState, uint64_t* sparkState, int min, int max)
{
	for(int i = min; i <= max; i++)
		if((mainState[i] & sparkState[i]) != (sparkState[i]))
			return NO;
			
	return YES;
}


cuDef void Join(uint64_t* main, uint64_t* delta, int dx, int dy, int min, int max)
{	
	for(int i = min; i <= max; i++)
	{	
		int idx = (i + dx + N) % N;
		
		if(dy < 0)
			main[idx] |= CirculateRight(delta[i], -dy);
		else
			main[idx] |= CirculateRight(delta[i], 64 -dy);
	}
}

cuDef uint64_t LocateAtX(uint64_t* state, int* xList, int* yList, int len, int x, int negate)
{
	uint64_t result = ~0ULL;
	
	for(int i = 0; i < len; i++)
	{
		int idx = (x + xList[i] + N) % N;
		int circulate = (yList[i] + 64) % 64;
		
		if(negate == NO)
			result &= CirculateRight(state[idx], circulate);
		else
			result &= ~CirculateRight(state[idx],circulate);
		
		if(result == 0ULL)
			break;
	}

	return result;
}

int State2Locator(uint64_t state[N], int* xList, int* yList)
{
	int curIdx = 0; 
	
	for(int j = 0; j < N; j++)
	{
        for(int i = 0; i < N; i++)
		{
            int val = Get(i, j, state);
			
			if(val == 1)
			{
				xList[curIdx] = i;
				yList[curIdx] = j;
				curIdx++;
			}
		}
	}
	
	return curIdx;
}

typedef struct 
{
	int x;
	int y; 
	int w; 
	int h;
	int s;

	int total;
		
} LifeIterator;


cuDef uint64_t* NewState(char* rle)
{
	uint64_t* result = HostNewArray<uint64_t>(N);
	Parse(result, rle);
	
	return result;
}


cuDef uint64_t* NewState(char* rle, int dx, int dy)
{
	uint64_t* result = HostNewArray<uint64_t>(N);
	Parse(result, rle);
	Move(result, dx, dy);
	
	return result;
}

cuDef LifeIterator* NewIterator(int x, int y, int w, int h, int s)
{
	LifeIterator* iter = HostNewPointer<LifeIterator>();
	
	iter -> x = x;
	iter -> y = y;
	iter -> w = w;
	iter -> h = h;
	iter -> s = s;
	iter -> total = w * h * s;
	
	return iter;
}

cuDef uint64_t* StatesContainer(uint64_t* state, int s)
{
	uint64_t* result = HostNewArray<uint64_t>(N * s);

	for(int i = 0; i < N * s; i++)
		result[i] = 0;

	Copy(result, state, COPY);

	for(int i = 1; i < s; i++)
	{
		Copy(&(result[N * i]), &(result[N * (i - 1)]), COPY);
		IterateState(&(result[N * i]), 0, N-1);		
	}
	
	return result;
	
}

cuDef LifeIterator* CopyToDevice(LifeIterator* cuIter)
{
	return  CudaCopyPointerToDevice<LifeIterator>(cuIter);
}

cuDef int3 FindState(LifeIterator* iter, uint64_t idx)
{
	int3 result; 
	
	result.x = idx % (iter -> w);
	idx = (idx - result.x) / (iter -> w);
	
	result.y = idx % (iter -> h);
	idx = (idx - result.y) / (iter -> h);
	
	if(idx < iter->s)
	{
		result.z = iter->s; 
	}
	else
	{
		result.z = -1;
	}
	
	result.x += iter->x;
	result.y += iter->y;
	
	return result;	
}

cuDef void FindState(LifeIterator* iter1, LifeIterator* iter2, uint64_t idx, int3 result[2])
{
	
	long val = idx % iter1-> total;
	
	result[0] = FindState(iter1, val);
	idx = (idx - val) / (iter1-> total);
	
	result[1] = FindState(iter2, idx);
	
}

cuDef void FindState(LifeIterator* iter1, LifeIterator* iter2, LifeIterator* iter3, uint64_t idx, int3 result[3])
{
	long val = idx % (iter1-> total);
	
	result[0] = FindState(iter1, val);
	
	idx = (idx - val) / (iter1-> total);
	val = idx % (iter2-> total);
	
	result[1] = FindState(iter2, val);
	
	idx = (idx - val) / (iter2-> total);
	result[2] = FindState(iter3, idx);
	
}

cuDef void Append(uint64_t* state, uint64_t* delta)
{
	Copy(state, delta, OR);
}

cuDef void Append(uint64_t* state, LifeIterator* iter, uint64_t* states, int3 val)
{
	if(val.z == -1)
		return;
		
	Join(state, &(states[val.z]) - 1, val.x, val.y, 0, N - 1);
}

cuDef void Append(uint64_t* state, LifeIterator* iter, uint64_t* states, uint64_t val)
{
	Append(state, iter, states, FindState(iter, val));
}

cuDef void Append(uint64_t* state, LifeIterator* iter1, LifeIterator* iter2, uint64_t* states1, uint64_t* states2, int3 val[2])
{
	if(val[1].z == -1)
		return;
	
	Append(state, iter1, states1,  val[0]);
	Append(state, iter2, states2,  val[1]);
}

cuDef void Append(uint64_t* state, LifeIterator* iter1, LifeIterator* iter2, uint64_t* states1, uint64_t* states2, uint64_t val)
{	
	int3 stateVal[2];
	FindState(iter1, iter2, val, stateVal);
	Append(state, iter1, iter2, states1, states2, stateVal);
}

cuDef void Append(uint64_t* state, LifeIterator* iter1, LifeIterator* iter2, LifeIterator* iter3, uint64_t* states1, uint64_t* states2, uint64_t* states3, int3 val[3])
{
	if(val[2].z == -1)
		return;
	
	Append(state, iter1, states1, val[0]);
	Append(state, iter2, states2, val[1]);
	Append(state, iter3, states3, val[2]);
}

cuDef void Append(uint64_t* state, LifeIterator* iter1, LifeIterator* iter2, LifeIterator* iter3, uint64_t* states1, uint64_t* states2, uint64_t* states3, uint64_t val)
{
	int3 stateVal[3];
	FindState(iter1, iter2, iter3, val, stateVal);
	Append(state, iter1, iter2, iter3, states1, states2, states3, stateVal);
}

cuDef long NumBlocks(LifeIterator* iter, int threads, int loops)
{
	return (long)((iter->total)  / (loops * threads)) + 1;
}

cuDef uint64_t NumBlocks(LifeIterator* iter1, LifeIterator* iter2, int threads, int loops)
{
	return (int)(((iter2->total) * (iter1->total)) / (loops * threads)) + 1;
}

cuDef uint64_t NumBlocks(LifeIterator* iter1, LifeIterator* iter2, LifeIterator* iter3, int threads, int loops)
{
	return (int)(((iter3->total) * (iter2->total) * (iter1->total)) / (loops * threads)) + 1;
}

cuDef uint64_t NumBlocks(LifeIterator* iter)
{
	return NumBlocks(iter, THREADS, ITER_PER_THREAD);
}

cuDef uint64_t NumBlocks(LifeIterator* iter1, LifeIterator* iter2)
{
	return NumBlocks(iter1, iter2, THREADS, ITER_PER_THREAD);
}

cuDef uint64_t NumBlocks(LifeIterator* iter1, LifeIterator* iter2, LifeIterator* iter3)
{
	return NumBlocks(iter1, iter2, iter3, THREADS, ITER_PER_THREAD);
}

cuDef uint64_t HashCode(uint64_t* state)
{
	uint64_t result = 0ull;

	for(int i = 0; i < N; i++)
		result ^= state[i];
	
	return result;

}

cuDef int IsFirstBigger(int3 i, int3 j)
{
	if(i.x > j.x)
		return YES;
	if(i.x < j.x)
		return NO;

	if(i.y > j.y)
		return YES;
	if(i.y < j.y)
		return NO;

	if(i.z > j.z)
		return YES;
	
	return NO;
}