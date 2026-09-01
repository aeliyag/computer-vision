import numpy as np
import hw1





def compute_gradients(image, scale = 1.0):
   """
   Compute image gradients using Sobel filters.

   Arguments:
      image  - a grayscale image in the form of a 2D numpy array
      scale  - scale factor

   Returns:
      ix     - a 2D numpy array with the x-gradients
      iy     - a 2D numpy array with the y -gradients
   """
   if scale > 0: 
      image_blur = hw1.denoise_gaussian(image, sigma = scale)
   else: 
      image_blur = image
   ix, iy = hw1.sobel_gradients(image_blur)
   return ix, iy 

def compute_second_moment_matrix(ix, iy, scale = 1.0):
   """
   Compute the components of the second moment matrix.

   Arguments:
      ix     - a 2D numpy array with the x-gradients
      iy     - a 2D numpy array with the y -gradients
      scale  - scale factor

   Returns:
      horz_grad    - a 2D numpy array with the Ix^2 values
      vert_grad    - a 2D numpy array with the Iy^2 values
      between_grad - a 2D numpy array with the Ix*Iy values
   """
   ix2 = ix * ix
   iy2 = iy * iy
   ixy = ix * iy
   horz_grad = hw1.denoise_gaussian(ix2, sigma = scale)
   vert_grad = hw1.denoise_gaussian(iy2, sigma = scale)
   between_grad = hw1.denoise_gaussian(ixy, sigma = scale)
   return horz_grad, vert_grad, between_grad

def harris_response(g_Ixx, g_Iyy, g_Ixy, alpha):
   """
   Compute the Harris corner response.

   Arguments:
      g_Ixx    - a 2D numpy array with the Ix^2 values
      g_Iyy    - a 2D numpy array with the Iy^2 values
      g_Ixy    - a 2D numpy array with the Ix*Iy values
      alpha        - between 0.04 - 0.06

   Returns:
      r            - a 2D numpy array of the Harris responses
   """
   det = g_Ixx * g_Iyy - g_Ixy * g_Ixy
   trace = g_Ixx + g_Iyy
   r = det - alpha * trace * trace
   return r

def nonmax_suppression(h_response, radius, threshold=0):
   """
   Apply non-maximum suppression to the Harris response map.

   Arguments:
      h_response  - a 2D numpy array of Harris response
      radius    - radius for non-maximum suppression
   Returns:
      keypoints - points after non-maximum suppression
   """
   h, w = h_response.shape 
   keypoints = []
   for i in range(h): 
      viewy1 = max(0, i - radius)
      viewy2 = min(h, i + radius + 1)

      for j in range(w): 
         val = h_response[i, j]
         if val > threshold: 
            viewx1 = max(0, j - radius)
            viewx2 = min(w, j + radius + 1)
            local_patch = h_response[viewy1:viewy2, viewx1:viewx2]
            if val == np.max(local_patch): 
               keypoints.append((j, i, val))
   return keypoints


def top_k_split(keypoints, k):
   """
   Selection sort to find teh top k keypoints and their values  
   
   :param keypoints: list of kepoints
   :param k: number opoints 
   :return: 
      xs: x coordinates of the keypoints
      ys: y coordinates of the keypoints
      scores: scores of the keypoints
   """

   kp = keypoints[:]
   xs, ys, scores = [], [], []
   k = min(k, len(kp))

   for _ in range(k):
      max_i = 0 
      max_val = kp[0][2]
      for i in range(1, len(kp)):
         if kp[i][2] > max_val: 
            max_val = kp[i][2]
            max_i = i
      xs.append(kp[max_i][0])
      ys.append(kp[max_i][1])
      scores.append(kp[max_i][2])
      kp.pop(max_i)
   return np.array(xs), np.array(ys), np.array(scores)


"""
   INTEREST POINT OPERATOR (12 Points Implementation + 3 Points Write-up)

   Implement an interest point operator of your choice.

   Your operator could be:

   (A) The Harris corner detector (Szeliski 7.1.1)

               OR

   (B) The Difference-of-Gaussians (DoG) operator defined in:
       Lowe, "Distinctive Image Features from Scale-Invariant Keypoints", 2004.
       https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

               OR

   (C) Any of the alternative interest point operators appearing in
       publications referenced in Szeliski or in lecture

              OR

   (D) A custom operator of your own design

   You implementation should return locations of the interest points in the
   form of (x,y) pixel coordinates, as well as a real-valued score for each
   interest point.  Greater scores indicate a stronger detector response.

   In addition, be sure to apply some form of spatial non-maximum suppression
   prior to returning interest points.

   Whichever of these options you choose, there is flexibility in the exact
   implementation, notably in regard to:

   (1) Scale

       At what scale (e.g. over what size of local patch) do you operate?

       You may optionally vary this according to an input scale argument.

       We will test your implementation at the default scale = 1.0, so you
       should make a reasonable choice for how to translate scale value 1.0
       into a size measured in pixels.

   (2) Nonmaximum suppression

       What strategy do you use for nonmaximum suppression?

       A simple (and sufficient) choice is to apply nonmaximum suppression
       over a local region.  In this case, over how large of a local region do
       you suppress?  How does that tie into the scale of your operator?

   For making these, and any other design choices, keep in mind a target of
   obtaining a few hundred interest points on the examples included with
   this assignment, with enough repeatability to have a large number of
   reliable matches between different views.

   If you detect more interest points than the requested maximum (given by
   the max_points argument), return only the max_points highest scoring ones.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      image       - a grayscale image in the form of a 2D numpy array
      max_points  - maximum number of interest points to return
      scale       - (optional, for your use only) scale factor at which to
                    detect interest points
      mask        - (optional, for your use only) foreground mask constraining
                    the regions to extract interest points
   Returns:
      xs          - numpy array of shape (N,) containing x-coordinates of the
                    N detected interest points (N <= max_points)
      ys          - numpy array of shape (N,) containing y-coordinates
      scores      - numpy array of shape (N,) containing a real-valued
                    measurement of the relative strength of each interest point
                    (e.g. corner detector criterion OR DoG operator magnitude)
"""
def find_interest_points(image, max_points = 200, scale = 1.0, mask = None):
   # check that image is grayscale
   assert image.ndim == 2, 'image should be grayscale'
   ##########################################################################
   ix, iy = compute_gradients(image, scale)
   g_Ixx, g_Iyy, g_Ixy = compute_second_moment_matrix(ix, iy, scale) 
   r = harris_response(g_Ixx, g_Iyy, g_Ixy, alpha = 0.04)
   radius = max(1, int(3 * scale))
   non_max = nonmax_suppression(r, radius)
   xs, ys, scores = top_k_split(non_max, max_points)
   ##########################################################################
   return xs, ys, scores


def compute_gradient_mag_ori(image, scale = 1.0):
   """
   Compute gradient magnitude and orientation.
   First smooths the image using a Gaussian filter at the given scale,
   then computes gradients using Sobel filters.
   square root of sum of squares of gradients gives magnitude,
   arctan of y-gradient over x-gradient gives orientation.

   Arguments:
      image    - a grayscale image in the form of a 2D numpy array
   Returns:
      mag      - a 2D numpy array with gradient magnitudes
      ori      - a 2D numpy array with gradient orientations
   """
   img = hw1.denoise_gaussian(image, scale) # reduce noise
   ix, iy = hw1.sobel_gradients(img) # compute derivatives
   mag = np.sqrt(ix * ix + iy * iy) # calc gradient magnitude 
   ori = np.arctan2(iy, ix) # calc gradient orientation
   return mag, ori 

def gaussian_windo(patch_width, sigma_factor = 0.5):
   """
   Create Gaussian window.
   recreating Gaussian weights

   Arguments:
      patch_width  - width of the patch
      sigma_factor - factor to compute sigma

   Returns:
      gauss_w      - a 2D numpy array with Gaussian weights
   """
   half = patch_width // 2 
   sigma = sigma_factor * patch_width
   gauss_window = np.zeros((patch_width, patch_width))
   for i in range(-half, half + 1):
      for j in range(-half, half + 1):
         gauss_window[i + half, j + half] = np.exp(-(i * i + j * j) / (2 * sigma * sigma))
   return gauss_window

def descriptor_at_point(mag, ori, x, y, patch_width,
                        cell_width, num_cels, num_bins, gauss_w):
   """
   Compute feature descriptor at a given point.

   Arguments:
      mag         - a 2D numpy array with gradient magnitudes
      ori         - a 2D numpy array with gradient orientations
      x           - x-coordinate of the interest point
      y           - y-coordinate of the interest point
      patch_width - width of the patch
      cell_width  - width of each cell
      num_cels    - number of cells per side
      num_bins    - number of orientation bins
      gauss_w     - a 2D numpy array with Gaussian weights
   Returns:
      feature     - a 1D numpy array with feature descriptor
   """
   x = int(x)
   y = int(y)
   half = patch_width // 2
   h, w = mag.shape
   feature = np.zeros((num_cels * num_cels * num_bins,))
   for i in range(-half, half + 1):
      for j in range(-half, half + 1):
         img_x = x + j
         img_y = y + i
         if img_x < 0 or img_x >= w or img_y < 0 or img_y >= h:
            continue
         mag_ij = mag[img_y, img_x]
         ori_ij = ori[img_y, img_x]
         weight = gauss_w[i + half, j + half]
         mag_weighted = mag_ij * weight

         cell_x = (j + half) // cell_width
         cell_y = (i + half) // cell_width
         if cell_x < 0 or cell_x >= num_cels or cell_y < 0 or cell_y >= num_cels:
            continue
         bin_idx = int(((ori_ij + np.pi) / (2 * np.pi)) * num_bins)
         bin_idx = bin_idx % num_bins

         feature_idx = (cell_y * num_cels + cell_x) * num_bins + bin_idx
         feature[feature_idx] += mag_weighted

   norm = np.linalg.norm(feature)
   if norm > 0:
      feature = feature / norm
   return feature





"""
   FEATURE DESCRIPTOR (12 Points Implementation + 3 Points Write-up)

   Implement a SIFT-like feature descriptor by binning orientation energy
   in spatial cells surrounding an interest point.

   Unlike SIFT, you do not need to build-in rotation or scale invariance.

   A reasonable default design is to consider a 3 x 3 spatial grid consisting
   of cell of a set width (see below) surrounding an interest point, marked
   by () in the diagram below.  Using 8 orientation bins, spaced evenly in
   [-pi,pi), yields a feature vector with 3 * 3 * 8 = 72 dimensions.

             ____ ____ ____
            |    |    |    |
            |    |    |    |
            |____|____|____|
            |    |    |    |
            |    | () |    |
            |____|____|____|
            |    |    |    |
            |    |    |    |
            |____|____|____|

                 |----|
                  width

   You will need to decide on a default spatial width.  Optionally, this can
   be a multiple of a scale factor, passed as an argument.  We will only test
   your code by calling it with scale = 1.0.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

  Arguments:
      image    - a grayscale image in the form of a 2D numpy
      xs       - numpy array of shape (N,) containing x-coordinates
      ys       - numpy array of shape (N,) containing y-coordinates
      scale    - scale factor

   Returns:
      feats    - a numpy array of shape (N,K), containing K-dimensional
                 feature descriptors at each of the N input locations
                 (using the default scheme suggested above, K = 72)
"""
def extract_features(image, xs, ys, scale = 1.0):
   # check that image is grayscale
   assert image.ndim == 2, 'image should be grayscale'
   ##########################################################################
   num_cels = 3 
   num_bins = 8 
   cell_width = max(1, int(3 * scale))
   patch_width = cell_width * num_cels
   mag, ori = compute_gradient_mag_ori(image, scale) 
   gauss_w = gaussian_windo(patch_width)

   features = np.zeros((xs.shape[0], num_cels * num_cels * num_bins)) 
   for i in range(xs.shape[0]):
      features[i,:] = descriptor_at_point(mag, ori, xs[i], ys[i], patch_width,
                                    cell_width, num_cels, num_bins, gauss_w)
   feats = features 
   ##########################################################################
   return feats

"""
   FEATURE MATCHING (7 Points Implementation + 3 Points Write-up)

   Given two sets of feature descriptors, extracted from two different images,
   compute the best matching feature in the second set for each feature in the
   first set.

   Matching need not be (and generally will not be) one-to-one or symmetric.
   Calling this function with the order of the feature sets swapped may
   result in different returned correspondences.

   For each match, also return a real-valued score indicating the quality of
   the match.  This score could be based on a distance ratio test, in order
   to quantify distinctiveness of the closest match in relation to the second
   closest match.  It could optionally also incorporate scores of the interest
   points at which the matched features were extracted.  You are free to
   design your own criterion.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      feats0   - a numpy array of shape (N0, K), containing N0 K-dimensional
                 feature descriptors (generated via extract_features())
      feats1   - a numpy array of shape (N1, K), containing N1 K-dimensional
                 feature descriptors (generated via extract_features())
      scores0  - a numpy array of shape (N0,) containing the scores for the
                 interest point locations at which feats0 was extracted
                 (generated via find_interest_point())
      scores1  - a numpy array of shape (N1,) containing the scores for the
                 interest point locations at which feats1 was extracted
                 (generated via find_interest_point())

   Returns:
      matches  - a numpy array of shape (N0,) containing, for each feature
                 in feats0, the index of the best matching feature in feats1
      scores   - a numpy array of shape (N0,) containing a real-valued score
                 for each match
"""
def match_features(feats0, feats1, scores0, scores1):
   ##########################################################################
   N0, feature_count = feats0.shape
   N1 = feats1.shape[0]
   matches = np.zeros((N0,), dtype=int) # need to specifiy type
   scores = np.zeros((N0,))

   for i in range(N0):
      NN1 = float('inf')
      NN2 = float('inf')
      best_index = -1 

      for j in range(N1):
         dist = 0
         for k in range(feature_count):
            diff = feats0[i, k] - feats1[j, k]
            dist += diff * diff
         if dist < NN1:
            NN2 = NN1
            NN1 = dist
            best_index = j
         elif dist < NN2:
            NN2 = dist
      matches[i] = best_index
      if NN2 == 0:
         scores[i] = 1 # this is a good feature, we want to include it 
      else:
         scores[i] = NN1 / NN2

   ##########################################################################
   return matches, scores


def compute_possible_translations(xs0, ys0, xs1, ys1, matches):
   """
   Compute possible translations based on matched features.
   """
   N0 = xs0.shape[0]
   potential_tx = np.zeros((N0,))
   potential_ty = np.zeros((N0,))

   for i in range(N0):
      match_index = int(matches[i])
      potential_tx[i] = xs1[match_index] - xs0[i]
      potential_ty[i] = ys1[match_index] - ys0[i]
   return potential_tx, potential_ty

def create_hough_grid(potential_tx, potential_ty, vote_width):
   """
   Create Hough grid for voting.
   """
   tx_min = min(potential_tx)
   tx_max = max(potential_tx)
   ty_min = min(potential_ty)
   ty_max = max(potential_ty)
   num_x_bins = int((tx_max - tx_min) / vote_width) + 1
   num_y_bins = int((ty_max - ty_min) / vote_width) + 1
   votes = []
   for _ in range(num_x_bins):
      votes.append([0] * num_y_bins)
   votes = np.array(votes)
   return votes, tx_min, ty_min

def cast_votes(votes, potential_tx, potential_ty, scores, tx_min, ty_min, vote_width):
   """
   Cast votes into Hough grid.
   """
   N = potential_tx.shape[0]
   for i in range(N):
      bin_x = int((potential_tx[i] - tx_min) / vote_width)
      bin_y = int((potential_ty[i] - ty_min) / vote_width)
      if scores[i] == 0: 
         continue 
      weight = 1.0 / scores[i]
      votes[bin_x, bin_y] += weight

def find_best_translation(votes):
   """
   Find best translation from Hough votes.
   """
   best_tx, best_ty = 0, 0
   max_votes = 0
   num_x_bins, num_y_bins = votes.shape
   for i in range(num_x_bins):
      for j in range(num_y_bins):
         if votes[i, j] > max_votes:
            max_votes = votes[i, j]
            best_tx = i
            best_ty = j
   return best_tx, best_ty

def find_inliers(potential_tx, potential_ty, best_tx, best_ty, 
                 vote_width, tx_min, ty_min):
   """
   Find inliers based on best translation.
   """
   inliers = []
   N = potential_tx.shape[0]
   for i in range(N):
      bin_x = int((potential_tx[i] - tx_min) / vote_width)
      bin_y = int((potential_ty[i] - ty_min) / vote_width)
      if bin_x == best_tx and bin_y == best_ty:
         inliers.append((potential_tx[i], potential_ty[i]))
   return inliers

def least_square(inliers):
   """
   Compute robust least square estimate of translation.
   """
   if len(inliers) == 0:
      return 0, 0
   sum_tx = 0
   sum_ty = 0
   for tx, ty in inliers:
      sum_tx += tx
      sum_ty += ty
   avg_tx = sum_tx / len(inliers)
   avg_ty = sum_ty / len(inliers)
   return avg_tx, avg_ty
"""
   HOUGH TRANSFORM (7 Points Implementation + 3 Points Write-up)

   Assuming two images of the same scene are related primarily by
   translational motion, use a predicted feature correspondence to
   estimate the overall translation vector t = [tx ty].

   Your implementation should use a Hough transform that tallies votes for
   translation parameters.  Each pair of matched features votes with some
   weight dependant on the confidence of the match; you may want to use your
   estimated scores to determine the weight.

   In order to accumulate votes, you will need to decide how to discretize the
   translation parameter space into bins.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      xs0     - numpy array of shape (N0,) containing x-coordinates of the
                interest points for features in the first image
      ys0     - numpy array of shape (N0,) containing y-coordinates of the
                interest points for features in the first image
      xs1     - numpy array of shape (N1,) containing x-coordinates of the
                interest points for features in the second image
      ys1     - numpy array of shape (N1,) containing y-coordinates of the
                interest points for features in the second image
      matches - a numpy array of shape (N0,) containing, for each feature in
                the first image, the index of the best match in the second
      scores  - a numpy array of shape (N0,) containing a real-valued score
                for each pair of matched features

   Returns:
      tx      - predicted translation in x-direction between images
      ty      - predicted translation in y-direction between images
      votes   - a matrix storing vote tallies; this output is provided for
                your own convenience and you are free to design its format
"""
def hough_votes(xs0, ys0, xs1, ys1, matches, scores):
   ##########################################################################
   vote_width = 5 
   potential_tx, potential_ty = compute_possible_translations(xs0, ys0, xs1, ys1, matches)
   votes, tx_min, ty_min = create_hough_grid(potential_tx, potential_ty, vote_width)
   cast_votes(votes, potential_tx, potential_ty, scores, tx_min, ty_min, vote_width)
   best_tx, best_ty = find_best_translation(votes)
   inliers = find_inliers(potential_tx, potential_ty, best_tx, best_ty, 
                          vote_width, tx_min, ty_min)
   tx, ty = least_square(inliers) 
   ##########################################################################
   return tx, ty, votes


def center_votes(xs_temp, ys_temp, xs_test, ys_test, matches, t_shape):
   """
   Center votes based on matched features.
   """
   h, w = t_shape 
   center_x = w / 2
   center_y = h / 2
   N = xs_temp.shape[0]
   valid_votes_x = []
   valid_votes_y = []
   for i in range(N):
      match_index = int(matches[i])
      if match_index < 0:
         continue
      vote_x = xs_test[match_index] - (xs_temp[i] - center_x)
      vote_y = ys_test[match_index] - (ys_temp[i] - center_y)
      valid_votes_x.append(vote_x)
      valid_votes_y.append(vote_y)
   return np.array(valid_votes_x), np.array(valid_votes_y)
def resize_image(image, scale):
   """
   Resize image by scale factor.

   """
   if scale == 1.0: 
      return image
   elif scale < 1.0: 
      inv_scale = 1.0 / scale
      return hw1.smooth_and_downsample(image, int(round(inv_scale)))
   else:
      return hw1.bilinear_upsampling(image, int(round(scale)))

"""
    OBJECT DETECTION (10 Points Implementation + 5 Points Write-up)

    Implement an object detection system which, given multiple object
    templates, localizes the object in the input (test) image by feature
    matching and hough voting.

    The first step is to match features between template images and test image.
    To prevent noisy matching from background, the template features should
    only be extracted from foreground regions.  The dense point-wise matching
    is then used to compute a bounding box by hough voting, where box center is
    derived from voting output and the box shape is simply the size of the
    template image.

    To detect potential objects with diversified shapes and scales, we provide
    multiple templates as input.  To further improve the performance and
    robustness, you are also REQUIRED to implement a multi-scale strategy
    either:
       (a) Implement multi-scale interest points and feature descriptors OR
       (b) Repeat a single-scale detection procedure over multiple image scales
           by resizing images.

    In addition to your implementation, include a brief write-up (in hw2.pdf)
    of your design choices on multi-scale implementaion and samples of
    detection results (please refer to display_bbox() function in visualize.py).

    Arguments:
        template_images - a list of gray scale images.  Each image is in the
                          form of a 2d numpy array which is cropped to tightly
                          cover the object.

        template_masks  - a list of binary masks having the same shape as the
                          template_image.  Each mask is in the form of 2d numpy
                          array specyfing the foreground mask of object in the
                          corresponding template image.

        test_img        - a gray scale test image in the form of 2d numpy array
                          containing the object category of interest.

    Returns:
         bbox           - a numpy array of shape (4,) specifying the detected
                          bounding box in the format of
                             (x_min, y_min, x_max, y_max)

"""
def object_detection(template_images, template_masks, test_img):
   ##########################################################################
   top_score = -1 
   best_bbox = None 
   best_inliers = 0 
   best_scale = None 
   best_template = None 

   scales = [1.0] 

   vote_width = 5

   max_points_test = 200
   max_points_temp = 100

   for s in scales: 
      test_s = resize_image(test_img, s)
      print(s, test_img.shape, test_s.shape)
      xs_test, ys_test, scores_test = find_interest_points(test_s, max_points_test, 1.0)
      feats_test = extract_features(test_s, xs_test, ys_test, 1.0)

      for t in range(len(template_images)):
         temp_img = template_images[t]
         temp_mask = template_masks[t]
         xs_temp, ys_temp, scores_temp = find_interest_points(temp_img, max_points_temp, 1.0, temp_mask)
         feats_temp = extract_features(temp_img, xs_temp, ys_temp, 1.0)
         if xs_temp.size == 0 or xs_test.size == 0:
            continue
         matches, match_scores = match_features(feats_temp, feats_test, scores_temp, scores_test)
         if feats_temp.shape[0] == 0 or feats_test.shape[0] == 0:
            continue 
         votes_x, votes_y = center_votes(xs_temp, ys_temp, xs_test, ys_test, matches, temp_img.shape)
         if votes_x.size == 0: 
            continue
         votes, tx_min, ty_min = create_hough_grid(votes_x, votes_y, vote_width)
         cast_votes(votes, votes_x, votes_y, match_scores, tx_min, ty_min, vote_width)
         best_tx, best_ty = find_best_translation(votes)
         inliers = find_inliers(votes_x, votes_y, best_tx, best_ty, 
                                vote_width, tx_min, ty_min)
         center_x, center_y = least_square(inliers)

         h, w = temp_img.shape
         x_min = int(center_x - w / 2)
         y_min = int(center_y - h / 2)
         x_max = int(center_x + w / 2)
         y_max = int(center_y + h / 2)

         bbox = np.array([x_min, y_min, x_max, y_max])
         if s != 1.0:
            bbox = (bbox / s).astype(int)


         score = len(inliers)
         if score > top_score:
            top_score = score
            best_bbox = bbox
            best_inliers = score
            best_scale = s
            best_template = t
   bbox = best_bbox
   print(f"[object_detection] best_inliers={best_inliers}, best_scale={best_scale}, best_template={best_template}")

   ##########################################################################
   return bbox
