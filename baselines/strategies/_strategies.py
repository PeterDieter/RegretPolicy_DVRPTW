import numpy as np
from environment import State
import sys
import math
import time


def log(obj, newline=True, flush=False):
    # Write logs to stderr since program uses stdout to communicate with controller
    sys.stderr.write(str(obj))
    if newline:
        sys.stderr.write('\n')
    if flush:
        sys.stderr.flush()

def _filter_instance(observation: State, mask: np.ndarray):
    res = {}

    for key, value in observation.items():
        if key == 'capacity':
            res[key] = value
            continue

        if key == 'duration_matrix':
            res[key] = value[mask]
            res[key] = res[key][:, mask]
            continue

        res[key] = value[mask]

    return res


def determineScore(customers_not_due, observation, masked_DM_Due, full_duration_matrix, numberTotalCustomers, discountList, epochs_to_go, dueDict, customers_due, customer_idx_due):
        # For each non due customer
    scoreList, idxList, numberList, nearestDueIdxList = [], [], [], []
    for customer in customers_not_due[1:]:
        customer_idx = observation['customer_idx'][customer]
        # Caculate number of customers which are closer then due customer
        minDistanceToDueCustomer = np.ma.min(masked_DM_Due[customer_idx,:]+masked_DM_Due[:,customer_idx])/2
        nearestDueCustomerIdx = np.ma.argmin(masked_DM_Due[customer_idx,:]+masked_DM_Due[:,customer_idx])
        
        closestCustomerIdx = np.ma.argmin(masked_DM_Due[customer_idx,1:])+1
        if customer_idx in customer_idx_due:
            scoreList.append(0)
            idxList.append(customer_idx)
            numberList.append(customer)
            nearestDueIdxList.append(nearestDueCustomerIdx)
            continue
        closestCustomer = np.where(customer_idx_due == closestCustomerIdx)[0][0]
        timeWarp1 = observation['time_windows'][customer][0] - observation['time_windows'][customers_due[closestCustomer]][1] - minDistanceToDueCustomer
        timeWarp2 = observation['time_windows'][customer][1] - observation['time_windows'][customers_due[closestCustomer]][0] - minDistanceToDueCustomer

        a = np.ma.where(full_duration_matrix[customer_idx,1:]+full_duration_matrix[1:, customer_idx]  < minDistanceToDueCustomer*2)
        distances_smaller_closestDueCustomer = (minDistanceToDueCustomer - 0.5*full_duration_matrix[customer_idx,1:][a] - 0.5*full_duration_matrix[1:, customer_idx][a])
        opportunities = np.sum(distances_smaller_closestDueCustomer)

        # Calculate the number of epochs the customer is still in the system
        number_of_epochs_until_due = math.floor(min(epochs_to_go,(observation['time_windows'][customer][1]-full_duration_matrix[0, customer_idx])/3600))


        # Get matrix of notdue customers
        customerList = []
        for k in range(1,number_of_epochs_until_due+1):
            customerList += dueDict[k]
        mask_NotDue = np.ones_like(full_duration_matrix)
        mask_NotDue[:, customerList] = 0
        mask_NotDue[customerList, :] = 0
        mask_NotDue[customerList, customerList] = 0
        masked_DM_NotDue = np.ma.MaskedArray(full_duration_matrix, mask=mask_NotDue)
        k = minDistanceToDueCustomer - masked_DM_NotDue[customer_idx,1:][np.ma.where(masked_DM_NotDue[customer_idx,1:]+masked_DM_NotDue[1:, customer_idx]  < minDistanceToDueCustomer*2)]
        q = (np.sum(k))/(np.mean(full_duration_matrix[:, :]))

        # Calculate probability that better customers will still arrive until customer is due. This is done with counterprobability
        prob = 1
        for epoch in range(number_of_epochs_until_due):
            probNotChosenEpoch = ((1-1/numberTotalCustomers)**discountList[epoch])
            prob = probNotChosenEpoch * prob
        prob = 1 - prob
        x = (prob**2)*opportunities
        x = x/np.ma.mean(full_duration_matrix[1:,1:])
        # If probability that better customer will arrive is below than 50%: release customer
        if timeWarp1 > 0 and timeWarp2 > 0:
            x = x * (1 + (min(timeWarp1, timeWarp2)/500))

        scoreList.append(x+0.01*q)
        idxList.append(customer_idx)
        numberList.append(customer)
        nearestDueIdxList.append(nearestDueCustomerIdx)

    return idxList, scoreList, numberList, nearestDueIdxList

def _sequentialRegret(observation: State, staticInfo, currentEpoch):
    # Determine customers which are due/not due
    customer_idx_due = np.copy(observation['customer_idx'][observation['must_dispatch']])
    customer_due = np.copy(np.arange(len(observation['customer_idx']))[observation['must_dispatch']])
    customer_idx_not_due = np.copy(observation['customer_idx'][~observation['must_dispatch']])
    customers_not_due = np.copy(np.arange(len(observation['customer_idx']))[~observation['must_dispatch']])
    # If no customer is due, we do not release any 
    if len(customer_idx_due) < 1:
        mask = np.zeros(len(observation['customer_idx'])).astype(np.bool8)
        mask[0] = True
        return _filter_instance(observation, mask)
    # If all customers are due, we release all customers
    elif len(customer_idx_not_due) < 2:
        return {
        **observation,
        'must_dispatch': np.ones_like(observation['must_dispatch']).astype(np.bool8)
    }
    # Else we perform our proposed heuristic
    else:
        startTime = time.time()
        full_duration_matrix = staticInfo["dynamic_context"]['duration_matrix']
        numberTotalCustomers = len(full_duration_matrix)
        # Get matrix of due customers
        mask_Due = np.ones_like(full_duration_matrix)
        mask_Due[:, customer_idx_due] = 0
        mask_Due[customer_idx_due, :] = 0
        masked_DM_Due = np.ma.MaskedArray(full_duration_matrix, mask=mask_Due)
        
        # Calculate a discount list which determines how many customers we expect in the next epochs. This number usually decreases with each epoch due to filtering of time windows
        epochs_to_go = staticInfo["end_epoch"]-currentEpoch-1
        discountList = []
        for epoch in range(currentEpoch+1,staticInfo["end_epoch"]+1):
            discountList.append(round(((3600 + epoch*3600 + np.mean(full_duration_matrix[0, 1:]) <= staticInfo["dynamic_context"]['time_windows'][:,1]).sum()-1)/(len(full_duration_matrix)-1)*100))
        

        # For each non due customer
        dueDict = {}
        for customer in customers_not_due[1:]:
            customer_idx = observation['customer_idx'][customer]

            # Calculate the number of epochs the customer is still in the system
            number_of_epochs_until_due = math.floor(min(epochs_to_go,(observation['time_windows'][customer][1]-full_duration_matrix[0, customer_idx])/3600))
            if number_of_epochs_until_due not in dueDict:
                dueDict[number_of_epochs_until_due] = [customer_idx]
            else:
                dueDict[number_of_epochs_until_due].append(customer_idx)




        # We release at least all customers that are due
        release_customer = list(np.copy(customer_due))
        threshhold = 0.08 # Theses parameters can be further tuned
        notFinished = True


        while notFinished:
            idxList, scoreList,numberList,nearestDueIdxList = determineScore(customers_not_due, observation, masked_DM_Due, full_duration_matrix, numberTotalCustomers, discountList, epochs_to_go, dueDict, release_customer, customer_idx_due)
            if len(idxList) > 0:
                scoreList, idxList,numberList,nearestDueIdxList = zip(*sorted(zip(scoreList,idxList, numberList, nearestDueIdxList)))
                if scoreList[0] < threshhold:
                    release_customer.append(numberList[0])
                    customers_not_due = customers_not_due[customers_not_due != numberList[0]]

                    customer_idx_due = np.append(customer_idx_due,idxList[0])
                    # # # # Get matrix of due customers
                    mask_Due = np.ones_like(full_duration_matrix)
                    mask_Due[:, customer_idx_due] = 0
                    mask_Due[customer_idx_due, :] = 0
                    masked_DM_Due = np.ma.MaskedArray(full_duration_matrix, mask=mask_Due)
                    
                    if time.time() - startTime > 100:
                        notFinished = False

                else:
                    notFinished = False
            else:
                notFinished = False

        mask = np.zeros(len(observation['customer_idx'])).astype(np.bool8)
        mask[release_customer] = True
        mask[0] = True
        return _filter_instance(observation, mask)


def _greedy(observation: State, rng: np.random.Generator):
    return {
        **observation,
        'must_dispatch': np.ones_like(observation['must_dispatch']).astype(np.bool8)
    }


def _lazy(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    mask[0] = True
    return _filter_instance(observation, mask)


def _random(observation: State, rng: np.random.Generator):
    mask = np.copy(observation['must_dispatch'])
    mask = (mask | rng.binomial(1, p=0.5, size=len(mask)).astype(np.bool8))
    mask[0] = True
    return _filter_instance(observation, mask)


STRATEGIES = dict(
    greedy=_greedy,
    lazy=_lazy,
    random=_random,
    probabilityRegretPolicy=_sequentialRegret
)
