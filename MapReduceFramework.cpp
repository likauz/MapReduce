#include <atomic>
#include <algorithm>
#include <iostream>
#include <semaphore.h>
#include "MapReduceClient.h"
#include "Barrier.h"
#include "MapReduceFramework.h"

#define NO_PROGRESS 0;

typedef void* JobHandle;

//this struct is used to store all the needed information of running thread meaning ThreadContext.
typedef struct {
    int threadID;
    pthread_t thread;
    Barrier* barrier;
    unsigned int total_shuffle = 0;
    unsigned int cur_shuffle = 0;
    std::atomic<unsigned long int>* atomic_counter_map;
    std::atomic<unsigned long int>* atomic_counter_reduce;
    std::vector<std::pair<K2*,V2*>> intermediate;
    const MapReduceClient* client;
    const std::vector<std::pair<K1 *, V1 *>>* inputVec;
    int threadsNum;
    std::vector<std::vector<std::pair<K2*,V2*>>>* shuffle;
    OutputVec* output;
    pthread_mutex_t* ourMutex;
    stage_t* stage;
} ThreadContext;

//this struct is used to store all the needed information of running job meaning allJob-info.
typedef struct {
    bool joinCall = false;
    pthread_mutex_t* ourMutex;
    int numOfThread;
    ThreadContext* context;
} allJob;

/**
 * it's the user's responsibility to call this function from map.
 * this function inserts key & value to context.
 * @param key
 * @param value
 * @param context
 */
void emit2 (K2* key, V2* value, void* context)
{
    auto thread_context = (ThreadContext*) context;
    std::pair<K2*,V2*> pair;
    pair.first = key;
    pair.second = value;
    (thread_context -> intermediate).push_back(pair);
}

/**
 * This function responsible to activate or de activate the given mutex by the bool sign
 * @param mutexThread - given mutex
 * @param to_lock = if true meaning we want to lock the mutex, if false we want to unlock the mutex
 */
void mutexLockWithErrors(pthread_mutex_t* mutexThread, bool to_lock)
{
    if(to_lock && pthread_mutex_lock(mutexThread))
    {
        std::cerr << "system error: error creating mutex\n";
        exit(1);
    }
    if(!to_lock && pthread_mutex_unlock(mutexThread))
    {
        std::cerr << "system error: error realising mutex\n";
        exit(1);
    }
}

/**
 * it's the user's responsibility to call this function from reduce.
 * this function inserts key & value into context.
 * @param key
 * @param value
 * @param context
 */
void emit3 (K3* key, V3* value, void* context)
{
    auto thread_context = (ThreadContext*) context;
    std::pair<K3*,V3*> pair;
    pair.first= key;
    pair.second= value;
    mutexLockWithErrors(thread_context -> ourMutex, true);
    thread_context->output->push_back(pair);
    mutexLockWithErrors(thread_context -> ourMutex, false);
}


/**
 * compare function between the keys of teo given pairs
 * @param pair1
 * @param pair2
 * @return true if key1 > key2 false otherwise
 */
bool compareK2(std::pair<K2*,V2*>  pair1, std::pair<K2*,V2*> pair2)
{
    return (*pair1.first) < (*pair2.first);
}

/**
 * This function goes throw each threadContext intermediateVector and gets the total max key.
 * @param context
 * @return a pointer to the maximum key in al the intermediateVectors
 */
K2* getMax(ThreadContext* context)
{
    K2* max = nullptr;
    for(int i = 0; i < context -> threadsNum; i++)
    {
        // check min for thr intermediate of contexts[i]
        if (((context + i) -> intermediate).empty())
        {
            continue;
        }
        // meaning if we still didn't initialize max, or we found a better one
        if(max == nullptr || (*max) < *(((context + i) -> intermediate).back().first))
        {
            max = ((context + i) -> intermediate).back().first;
        }
    }
    return max;
}

/**
 * this function extract a vector of all the pairs with the given key from the
 * given context intermediateVector
 * @param context
 * @param now_key
 * @return
 */
std::vector<std::pair<K2*,V2*>> extract_vector(ThreadContext* context, K2* now_key)
{
    std::vector<std::pair<K2*,V2*>> result;
    while(!(context -> intermediate.empty()) && !(*(context -> intermediate.back().first) < (*now_key)))
    {
        result.push_back(context -> intermediate.back());
        (context -> intermediate).pop_back();
    }
    return result;
}

/**
 * This function is responsible for making the shuffle stage.
 * only thread 0 will enter here.
 * @param context
 */
void shuffle(ThreadContext* context)
{
    // changing the total amount of shuffle for percentage
    while(getMax(context) != nullptr) // pass over all the char in the intermediate
    {
        K2* now_key = getMax(context);
        std::vector<std::pair<K2*,V2*>> per_char;
        for(int i = 0; i < context -> threadsNum; i++)
        {
            std::vector<std::pair<K2*,V2*>> per_thread = extract_vector(context + i, now_key);
            if(!per_thread.empty())
            {
                per_char.insert(per_char.end(), per_thread.begin(), per_thread.end());
            }
        }
        (*(context -> shuffle)).push_back(per_char);
        
        // update cur_shuffle for percentage 
        mutexLockWithErrors(context -> ourMutex, true);
        context -> cur_shuffle += per_char.size();
        mutexLockWithErrors(context -> ourMutex, false);
    }
}

/**
 * This is the main function of a thread, meaning a thread will initialized with address to this function.
 * from here a thread going throw map stage -> sort -> shuffle -> reduce
 * @param arg : ThreadContext* representing given Thread
 * @return
 */
void* runAllThread(void* arg)
{
    auto context = (ThreadContext*) arg;
    // run map
    mutexLockWithErrors(context -> ourMutex, true);
    if ((*context -> stage) == UNDEFINED_STAGE)
    {
        (*context -> stage) = MAP_STAGE;
    }
    mutexLockWithErrors(context -> ourMutex, false);

    unsigned int i = (*(context->atomic_counter_map))++;
    while(i < (*(context->inputVec)).size())
    {
        std::pair<K1*,V1*> value = (*(context -> inputVec))[i];
        context -> client-> map(value.first, value.second, context);
        i = (*(context->atomic_counter_map))++;
    }

    // run sort
    std::sort(context->intermediate.begin(), context->intermediate.end(), compareK2);

    // barrier before shuffle
    context -> barrier-> barrier();

    // run shuffle
    if(context -> threadID == 0)
    {
        mutexLockWithErrors(context -> ourMutex, true);
        (*context -> stage) = SHUFFLE_STAGE;
        for(int j = 0; j < context -> threadsNum; j++)
        {
            context -> total_shuffle += context[j].intermediate.size();
        }
        mutexLockWithErrors(context -> ourMutex, false);
        shuffle(context);
        mutexLockWithErrors(context -> ourMutex, true);
        (*context -> stage) = REDUCE_STAGE;
        mutexLockWithErrors(context -> ourMutex, false);
    }

    // barrier before reduce
    context -> barrier -> barrier();

    // run reduce
    unsigned int index = (*(context->atomic_counter_reduce))++;
    while(index < (*(context->shuffle)).size())
    {
        std::vector<std::pair<K2*,V2*>> value = (*(context -> shuffle))[index];
        context -> client-> reduce(&value, context);
        index = (*(context->atomic_counter_reduce))++;
    }

    // finish
    return nullptr;
}

/**
 * The user will call this function in order to start the process
 * @param client : Map function & reduce function
 * @param inputVec : given input to start mapping
 * @param outputVec : a vector to give emit 3 meaning store the output values
 * @param multiThreadLevel : num of threads the user want to handle the process
 * @return : jobHandle, an object with in the user can get information about the running process
 */
JobHandle startMapReduceJob(const MapReduceClient& client,const InputVec& inputVec, OutputVec& outputVec,int multiThreadLevel)
{
    // initialized the shared element
    auto ourJob = new allJob;
    auto atomic_counter1 = new std::atomic<unsigned long int>(0);
    auto atomic_counter2 = new std::atomic<unsigned long int>(0);
    auto barrier = new Barrier (multiThreadLevel);
    auto shuffle = new std::vector<std::vector<std::pair<K2*,V2*>>>;
    ourJob -> context = new ThreadContext[multiThreadLevel];
    ourJob -> ourMutex = new pthread_mutex_t;
    ourJob -> numOfThread = multiThreadLevel;
    auto stage = new stage_t(UNDEFINED_STAGE);
    if(pthread_mutex_init(ourJob->ourMutex, nullptr))
    {
        std::cerr << "system error: error creating mutex\n";
        exit(1);
    }

    // initialized each thread and calling map step and sort
    for( int i = 0; i < multiThreadLevel; i++)
    {
        ourJob -> context[i].threadID = i;
        ourJob -> context[i].barrier = barrier;
        ourJob -> context[i].atomic_counter_map = atomic_counter1;
        ourJob -> context[i].atomic_counter_reduce = atomic_counter2;
        ourJob -> context[i].inputVec = &inputVec;
        ourJob -> context[i].client = &client;
        ourJob -> context[i].threadsNum = multiThreadLevel;
        ourJob -> context[i].shuffle = shuffle;
        ourJob -> context[i].output = &outputVec;
        ourJob -> context[i].ourMutex = ourJob->ourMutex;
        ourJob -> context[i].stage = stage;
        if (pthread_create(&ourJob -> context[i].thread, NULL, runAllThread, ourJob -> context + i) != 0)
        {
            std::cerr << "system error: error in pthread_create\n" << std::endl;
            exit(1);
        }
    }
    return ourJob;
}

/**
 * the user will send the JobHandle he got from startMapReduceJob.
 * the function will return only when the process finished meaning the outputVector full with the fit values.
 * p_join can only called once for each thread
 * @param job
 */
void waitForJob(JobHandle job)
{
    auto curJob =(allJob*) job;
    if(curJob->joinCall)
    {
        return;
    }
    for(int i=0;i<curJob->numOfThread;i++)
    {
        if(pthread_join(curJob->context[i].thread, nullptr) != 0)
        {
            std::cerr << "system error: error in pthread_join\n" << std::endl;
            exit(1);
        }
    }
    curJob->joinCall = true;
}

/**
 * the user will send the JobHandle he got from startMapReduceJob,
 * and will received in state the state of rhe job: stage and percentage.
 * @param job
 * @param state
 */
void getJobState(JobHandle job, JobState* state)
{
    float cur;
    float total;
    auto curJob =(allJob*) job;
    mutexLockWithErrors(curJob -> ourMutex, true);
    state -> stage = (*(curJob -> context[0].stage));
    switch (state -> stage) {
        case UNDEFINED_STAGE:
            state -> percentage = NO_PROGRESS;
            break;
        
        case MAP_STAGE:
            total = (float) (curJob -> context[0]).inputVec->size();
            cur = (float) *(curJob -> context[0]).atomic_counter_map;
            state -> percentage = 100;
            if(cur < total)
            {
                state -> percentage = 100 * cur / total;
            }
            break;

        case SHUFFLE_STAGE:
            state -> percentage = 100 * (float) (curJob -> context[0]).cur_shuffle /
                    (float)(curJob -> context[0]).total_shuffle;
            break;

        case REDUCE_STAGE:
            cur = (float) *(curJob -> context[0]).atomic_counter_reduce;
            total = (float) (curJob -> context[0]).shuffle->size();
            state -> percentage = 100;
            if(cur < total)
            {
                state -> percentage = 100 * cur / total;
            }
            break;

        default:
            break;
    }
    mutexLockWithErrors(curJob -> ourMutex, false);
}

/**
 * the user will send the JobHandle he got from startMapReduceJob,
 * when the jon finished the function will be responsible to free the memory allocation
 * @param job
 */
void closeJobHandle(JobHandle job)
{
    auto curJob =(allJob*) job;
    if(!curJob->joinCall)
    {
        waitForJob(job);
    }
    delete curJob->context[0].barrier;
    delete curJob->context[0].atomic_counter_map;
    delete curJob->context[0].atomic_counter_reduce;
    delete curJob->context[0].shuffle;
    delete curJob->context[0].stage;
    delete [] curJob -> context;
    delete curJob -> ourMutex;
    delete curJob;
}