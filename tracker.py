# Copyright (c) 2020, 2026 Mahdi Kazemi Moghaddam
# SPDX-License-Identifier: GPL-3.0-or-later
# Public-release preparation and portability changes: 2026-11-19.
"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import numpy as np
from scipy.optimize import linear_sum_assignment

def linear_assignment(cost):
  rows, columns = linear_sum_assignment(cost)
  return np.column_stack((rows, columns))
from filterpy.kalman import KalmanFilter
import math

def eu_distance(point_1, point_2):
  """
  Computes Euclidean distance between two ball points in the form [x1,y1,x2,y2]
  """
  distance = math.sqrt(math.pow((point_1[0] - point_2[0]), 2) + math.pow((point_1[1] - point_2[1]), 2))
  
  return distance

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2]-bbox[0]
  h = bbox[3]-bbox[1]
  x = bbox[0]+w/2.
  y = bbox[1]+h/2.
  s = w*h    #scale is just area
  r = w/float(h)
  return np.array([x,y,s,r]).reshape((4,1))

def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2]*x[3])
  h = x[2]/w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model (z is the size of measurement vector (top_left and botom_right coords) and 
    # ... x is the size of the state vector). Here 7: x_topl, y_topl, scale, ratio
    self.kf = KalmanFilter(dim_x=4, dim_z=2)
    # Motion model matrix (constant velocity)
    self.kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
    #  update matrix
    self.kf.H = np.array([[1,0,0,0],[0,1,0,0]])
    # Measurment noise matrix
    self.kf.R[:2, :2] *= 10
    self.kf.R[2:,2:] *= 10
    # Initial covariance matrix
    self.kf.P[2:,2:] *= 50. #give high uncertainty to the unobservable initial velocities
    self.kf.P[:2, :2] *= 50.
    # Process noise matrix
    self.kf.Q[-1,-1] *= 0.001
    self.kf.Q[2:,2:] *= 0.001

    #self.kf.x[:4] = convert_bbox_to_z(bbox)
    #print(bbox, np.array([bbox[0], bbox[1]]))
    self.kf.x[:2] = np.array([bbox[0], bbox[1]]).reshape((2,1))
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(np.array([bbox[0], bbox[1]]).reshape((2,1)))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    #if((self.kf.x[6] + self.kf.x[2]) <= 0):
    #  self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(self.kf.x[:2])
    #print(self.history)
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return np.array(self.kf.x[:2]).reshape((1, 2))

def associate_detections_to_trackers(detections, trackers, dist_threshold = 40):
  """
  Assigns detections to tracked object (both represented as points) based on the Euclidean distance
  detections and trackers are array: n*3=[[x, y, score], ...]
  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  #print('inside the association: ', detections, trackers)
  if(len(trackers) == 0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,3),dtype=int)
  
  iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  # Calculate the pair-wise Euclidean distance of all the current detections with current tracked objects
  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      iou_matrix[d,t] = eu_distance(det,trk)

  # Matches the current detections with current trackers based on Hungarian matching (lowest distance get matched together)
  matched_indices = linear_assignment(iou_matrix)
  #print('matched indices: ', matched_indices)

  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  
  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with high distance
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0],m[1]] > dist_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  
  if(len(matches) == 0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)



class Sort(object):
  def __init__(self,max_age=5,min_hits=1):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0

  def update(self,dets):
    """
    Params:
      dets - a numpy array of detections in the format [[x,y,score],[x,y,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns rows [x, y, track_id].

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    #get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers),3))
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict()
      #print(self.trackers[t].predict(), pos)
      trk[:] = [float(pos[0, 0]), float(pos[1, 0]), 0]
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    
    for t in reversed(to_del):
      self.trackers.pop(t)
    
    # three lists of indices of the detection and trackers for matched and unmatched
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

    #update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
      if(t not in unmatched_trks):
        d = matched[np.where(matched[:,1] == t)[0], 0]
        #print('dets: ', dets[d,:])
        trk.update(dets[d,:][0])

    #create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:]) 
        self.trackers.append(trk)
    
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()
        if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
          #print('d shape: ', d.shape)
          ret.append(np.concatenate((d, [[trk.id+1]]), axis=1).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        #remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    
    if(len(ret)>0):
      return np.concatenate(ret)

    return np.empty((0,3))
    
