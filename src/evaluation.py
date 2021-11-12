#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import torch
import numpy as np
from scipy.signal import find_peaks
from collections import defaultdict, Counter


def evaluate_overlap(evaluation_pairs):
    
    hits = 0
    counts = 0
    for targets,preds in tqdm(evaluation_pairs):
        assert len(targets)==len(preds)
        hits += sum(np.array(targets)==np.array(preds))
        counts += len(targets)
        
    return hits/counts


'''
Code for precision, recall, F1, R-value was adapted from unsupseg: https://github.com/felixkreuk/UnsupSeg
'''

def get_metrics(precision_counter, recall_counter, pred_counter, gt_counter):
    eps = 1e-7

    precision = precision_counter / (pred_counter + eps)
    recall = recall_counter / (gt_counter + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    os = recall / (precision + eps) - 1
    r1 = np.sqrt((1 - recall) ** 2 + os ** 2)
    r2 = (-os + recall - 1) / (np.sqrt(2))
    rval = 1 - (np.abs(r1) + np.abs(r2)) / 2

    return precision, recall, f1, rval


def get_stats(y, y_ids, yhat, yhat_ids, tolerance=0.02):

    precision_counter = 0
    recall_counter = 0
    pred_counter = 0
    gt_counter = 0

    for yhat_i, yhat_id in zip(yhat,yhat_ids):
        diff = np.abs(y - yhat_i)
        min_dist = diff.min()
        min_pos = np.argmin(diff)
        intersect = y_ids[min_pos].intersection(yhat_id)
        if len(intersect)>0:
            precision_counter += (min_dist <= tolerance)

    for y_i,y_id in zip(y,y_ids):
        diff = np.abs(yhat - y_i)
        min_dist = diff.min()
        min_pos = np.argmin(diff)
        intersect = yhat_ids[min_pos].intersection(y_id)
        if len(intersect)>0:
            recall_counter += (min_dist <= tolerance)

    pred_counter += len(yhat)
    gt_counter += len(y)

    p, r, f1, rval = get_metrics(precision_counter,
                                      recall_counter,
                                      pred_counter,
                                      gt_counter)
    return p, r, f1, rval

def get_all_stats(evaluation_pairs,tolerance=0.02):
    
    precision_counter = 0
    recall_counter = 0
    pred_counter = 0
    gt_counter = 0
    
    for (y,y_ids), (yhat, yhat_ids) in tqdm(evaluation_pairs):
        
        for yhat_i, yhat_id in zip(yhat,yhat_ids):
            diff = np.abs(y - yhat_i)
            min_dist = diff.min()
            min_pos = np.argmin(diff)
            intersect = y_ids[min_pos].intersection(yhat_id)
            if len(intersect)>0:
                precision_counter += (min_dist <= tolerance)

        for y_i,y_id in zip(y,y_ids):
            diff = np.abs(yhat - y_i)
            min_dist = diff.min()
            min_pos = np.argmin(diff)
            intersect = yhat_ids[min_pos].intersection(y_id)
            if len(intersect)>0:
                recall_counter += (min_dist <= tolerance)

        pred_counter += len(yhat)
        gt_counter += len(y)
        
    p, r, f1, rval = get_metrics(precision_counter,
                                      recall_counter,
                                      pred_counter,
                                      gt_counter)
    return p, r, f1, rval
 

   
def get_all_stats_boundary_only(evaluation_pairs,tolerance=0.02):
    
    precision_counter = 0
    recall_counter = 0
    pred_counter = 0
    gt_counter = 0
    
    for (y,y_ids), (yhat, yhat_ids) in tqdm(evaluation_pairs):
        
        for yhat_i, yhat_id in zip(yhat,yhat_ids):
            diff = np.abs(y - yhat_i)
            min_dist = diff.min()
            min_pos = np.argmin(diff)
            precision_counter += (min_dist <= tolerance)

        for y_i,y_id in zip(y,y_ids):
            diff = np.abs(yhat - y_i)
            min_dist = diff.min()
            min_pos = np.argmin(diff)
            recall_counter += (min_dist <= tolerance)

        pred_counter += len(yhat)
        gt_counter += len(y)
        
    p, r, f1, rval = get_metrics(precision_counter,
                                      recall_counter,
                                      pred_counter,
                                      gt_counter)
    return p, r, f1, rval
    

def detect_peaks(xi,prominence=0.1, width=None, distance=None):
    """detect peaks of next_frame_classifier
    
    Arguments:
        x {Array} -- a sequence of cosine distances
    """ 
    
       # shorten to actual length
    xmin, xmax = xi.min(), xi.max()
    xi = (xi - xmin) / (xmax - xmin)
    peaks, _ = find_peaks(xi, prominence=prominence, width=width, distance=distance)

    if len(peaks) == 0:
        peaks = np.array([len(xi)-1])


    return peaks




def detect_peaks_in_batch(x, prominence=0.1, width=None, distance=None):
    """detect peaks of next_frame_classifier
    
    Arguments:
        x {Array} -- batch of coscine distances per time
    """ 
    out = []

    for xi in x:
        if type(xi) == torch.Tensor:
            xi = xi.cpu().detach().numpy()  # shorten to actual length
        xmin, xmax = xi.min(), xi.max()
        xi = (xi - xmin) / (xmax - xmin)
        peaks, _ = find_peaks(xi, prominence=prominence, width=width, distance=distance)

        if len(peaks) == 0:
            peaks = np.array([len(xi)-1])

        out.append(peaks)

    return out


class PrecisionRecallMetric:
    def __init__(self):
        self.precision_counter = 0
        self.recall_counter = 0
        self.pred_counter = 0
        self.gt_counter = 0
        self.eps = 1e-5
        self.data = []
        self.tolerance = 0.02
        self.prominence_range = np.arange(0, 0.15, 0.1)
        self.width_range = [None, 1]
        self.distance_range = [None, 1]
        self.resolution = 1/49

    def get_metrics(self, precision_counter, recall_counter, pred_counter, gt_counter):
        EPS = 1e-7
        
        precision = precision_counter / (pred_counter + self.eps)
        recall = recall_counter / (gt_counter + self.eps)
        f1 = 2 * (precision * recall) / (precision + recall + self.eps)
        
        os = recall / (precision + EPS) - 1
        r1 = np.sqrt((1 - recall) ** 2 + os ** 2)
        r2 = (-os + recall - 1) / (np.sqrt(2))
        rval = 1 - (np.abs(r1) + np.abs(r2)) / 2

        return precision, recall, f1, rval

    def zero(self):
        self.data = []

    def update(self, seg, pos_pred,):
        for seg_i, pos_pred_i in zip(seg, pos_pred):
            self.data.append((seg_i, pos_pred_i))

    def get_stats_and_search_params(self, width=None, prominence=None, distance=None):
        print(f"calculating metrics using {len(self.data)} entries")
        max_rval = -float("inf")
        best_params = None
        segs = list(map(lambda x: x[0], self.data))
        yhats = list(map(lambda x: x[1], self.data))

        width_range = self.width_range
        distance_range = self.distance_range
        prominence_range = self.prominence_range

        # when testing, we would override the search with specific values from validation
        if prominence is not None:
            width_range = [width]
            distance_range = [distance]
            prominence_range = [prominence]

        for width in width_range:
            for prominence in prominence_range:
                for distance in distance_range:
                    precision_counter = 0
                    recall_counter = 0
                    pred_counter = 0
                    gt_counter = 0
                    peaks = detect_peaks_in_batch(yhats,
                                         prominence=prominence,
                                         width=width,
                                         distance=distance)

                    for (y, yhat) in zip(segs, peaks):
                        for yhat_i in yhat:
                            min_dist = np.abs(y - yhat_i*self.resolution).min()
                            precision_counter += (min_dist <= self.tolerance)
                        for y_i in y:
                            min_dist = np.abs(yhat*self.resolution - y_i).min()
                            recall_counter += (min_dist <= self.tolerance)
                        pred_counter += len(yhat)
                        gt_counter += len(y)

                    p, r, f1, rval = self.get_metrics(precision_counter,
                                                      recall_counter,
                                                      pred_counter,
                                                      gt_counter)
                    if rval > max_rval:
                        max_rval = rval
                        best_params = width, prominence, distance
                        out = (p, r, f1, rval)
        self.zero()
        print(f"best peak detection params: {best_params} (width, prominence, distance)")
        return out, best_params
    
    def get_stats(self, y, yhat):
        
        precision_counter = 0
        recall_counter = 0
        pred_counter = 0
        gt_counter = 0
        
    
        for yhat_i in yhat:
            min_dist = np.abs(y - yhat_i).min()
            precision_counter += (min_dist <= self.tolerance)
        for y_i in y:
            min_dist = np.abs(yhat - y_i).min()
            recall_counter += (min_dist <= self.tolerance)
        pred_counter += len(yhat)
        gt_counter += len(y)

        p, r, f1, rval = self.get_metrics(precision_counter,
                                          recall_counter,
                                          pred_counter,
                                          gt_counter)
        return p, r, f1, rval
