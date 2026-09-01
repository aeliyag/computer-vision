import numpy as np

"""
   Mirror an image about its border.

   Arguments:
      image - a 2D numpy array of shape (sx, sy)
      wx    - a scalar specifying width of the top/bottom border
      wy    - a scalar specifying width of the left/right border

   Returns:
      img   - a 2D numpy array of shape (sx + 2*wx, sy + 2*wy) containing
              the original image centered in its interior and a surrounding
              border of the specified width created by mirroring the interior
"""
def mirror_border(image, wx = 1, wy = 1):
   assert image.ndim == 2, 'image should be grayscale'
   sx, sy = image.shape
   # mirror top/bottom
   top    = image[:wx:,:]
   bottom = image[(sx-wx):,:]
   img = np.concatenate( \
      (top[::-1,:], image, bottom[::-1,:]), \
      axis=0 \
   )
   # mirror left/right
   left  = img[:,:wy]
   right = img[:,(sy-wy):]
   img = np.concatenate( \
      (left[:,::-1], img, right[:,::-1]), \
      axis=1 \
   )
   return img

"""
   Pad an image with zeros about its border.

   Arguments:
      image - a 2D numpy array of shape (sx, sy)
      wx    - a scalar specifying width of the top/bottom border
      wy    - a scalar specifying width of the left/right border

   Returns:
      img   - a 2D numpy array of shape (sx + 2*wx, sy + 2*wy) containing
              the original image centered in its interior and a surrounding
              border of zeros
"""
def pad_border(image, wx = 1, wy = 1):
   assert image.ndim == 2, 'image should be grayscale'
   sx, sy = image.shape
   img = np.zeros((sx+2*wx, sy+2*wy))
   img[wx:(sx+wx),wy:(sy+wy)] = image
   return img

"""
   Remove the border of an image.

   Arguments:
      image - a 2D numpy array of shape (sx, sy)
      wx    - a scalar specifying width of the top/bottom border
      wy    - a scalar specifying width of the left/right border

   Returns:
      img   - a 2D numpy array of shape (sx - 2*wx, sy - 2*wy), extracted by
              removing a border of the specified width from the sides of the
              input image
"""
def trim_border(image, wx = 1, wy = 1):
   assert image.ndim == 2, 'image should be grayscale'
   sx, sy = image.shape
   img = np.copy(image[wx:(sx-wx),wy:(sy-wy)])
   return img

"""
   Return an approximation of a 1-dimensional Gaussian filter.

   The returned filter approximates:

   g(x) = 1 / sqrt(2 * pi * sigma^2) * exp( -(x^2) / (2 * sigma^2) )

   for x in the range [-3*sigma, 3*sigma]
"""
def gaussian_1d(sigma = 1.0):
   width = np.ceil(3.0 * sigma)
   x = np.arange(-width, width + 1)
   g = np.exp(-(x * x) / (2 * sigma * sigma))
   g = g / np.sum(g)          # normalize filter to sum to 1 ( equivalent
   g = np.atleast_2d(g)       # to multiplication by 1 / sqrt(2*pi*sigma^2) )
   return g

"""
   CONVOLUTION IMPLEMENTATION (10 Points)

   Convolve a 2D image with a 2D filter.

   Requirements:

   (1) Return a result the same size as the input image.

   (2) You may assume the filter has odd dimensions.

   (3) The result at location (x,y) in the output should correspond to
       aligning the center of the filter over location (x,y) in the input
       image.

   (4) When computing a product at locations where the filter extends beyond
       the defined image, treat missing terms as zero.  (Equivalently stated,
       treat the image as being padded with zeros around its border).

   You must write the code for the nested loops of the convolutions yourself,
   using only basic loop constructs, array indexing, multiplication, and
   addition operators.  You may not call any Python library routines that
   implement convolution.

   Arguments:
      image  - a 2D numpy array
      filt   - a 1D or 2D numpy array, with odd dimensions
      mode   - 'zero': preprocess using pad_border or 'mirror': preprocess using mirror_border.

   Returns:
      result - a 2D numpy array of the same shape as image, containing the
               result of convolving the image with filt
"""
def conv_2d(image, filt, mode='zero'):
   assert image.ndim == 2, 'image should be grayscale'
   filt = np.atleast_2d(filt)

   img_x, img_y = image.shape 
   filt_x, filt_y = filt.shape 
   result = np.zeros((img_x, img_y))

   pad_y = filt_y // 2  # determine padding from filter size
   pad_x = filt_x // 2

   if mode == 'zero':
      img = pad_border(image, pad_x, pad_y)
   elif mode == 'mirror':
      img = mirror_border(image, pad_x, pad_y)
   
   for x in range(img_x):
      for y in range(img_y):
         value = 0 
         for f_x in range(filt_x):
            for f_y in range(filt_y):
               value += img[x + f_x, y + f_y] * filt[f_x, f_y]
         result[x][y] = value

   return result

"""
   GAUSSIAN DENOISING (5 Points)

   Denoise an image by convolving it with a 2D Gaussian filter.

   Convolve the input image with a 2D filter G(x,y) defined by:

   G(x,y) = 1 / sqrt(2 * pi * sigma^2) * exp( -(x^2 + y^2) / (2 * sigma^2) )

   You may approximate the G(x,y) filter by computing it on a
   discrete grid for both x and y in the range [-3*sigma, 3*sigma].

   See the gaussian_1d function for reference.

   Note:
   (1) Remember that the Gaussian is a separable filter.
   (2) Denoising should not create artifacts along the border of the image.
       Make an appropriate assumption in order to obtain visually plausible
       results along the border.

   Arguments:
      image - a 2D numpy array
      sigma - standard deviation of the Gaussian

   Returns:
      img   - denoised image, a 2D numpy array of the same shape as the input
"""
def denoise_gaussian(image, sigma = 1.0):
   g_x = gaussian_1d(sigma)
   g_T = g_x.T
   img = conv_2d(image, g_x, 'mirror')
   img = conv_2d(img, g_T, 'mirror')

   return img

"""
   CONVOLUTION VIA FFT (10 Points)

   Convolve a 2D image with a 2D filter using the convolution theorem
   using zero-padding.

   Convolution in the spatial domain corresponds to multiplication in 
   the frequency domain:

   F[I * K] = F[I]F[K]

   You may use NumPy's FFT (np.fft) routines to compute the Fourier transform, 
   but implement supporting logic yourself.

   Note

   (1) DFFT computes *circular convolution* by default, since it represents signals 
       as periodic. To obtain the same result as spatial convolution with zero padding, 
       you must explicitly pad the input image and the filter.

   (2) FFT assumes that the filter is located at the origin (0,0). Therefore, 
       before taking the FFT of the padded filter, you must shift it. You may use
       helpers from np.fft here.

   (3) After the inverse FFT, you must extract the correct cropping to return the
       desired convolution.

   Arguments:
      image  - a 2D numpy array
      filt   - a 1D or 2D numpy array, with odd dimensions

   Returns:
      result - a 2D numpy array of the same shape as image, containing the
               result of convolving the image with filt
"""
def conv_2d_fft(image, filt):
    assert image.ndim == 2
    filt = np.atleast_2d(filt)

    img_h, img_w = image.shape
    filt_h, filt_w = filt.shape

    padd_h = img_h + filt_h - 1
    padd_w = img_w + filt_w - 1

    pad_img = np.zeros((padd_h, padd_w))
    pad_img[:img_h, :img_w] = image

    pad_filt = np.zeros((padd_h, padd_w))
    pad_filt[:filt_h, :filt_w] = np.fft.ifftshift(filt) 

    f_i = np.fft.fft2(pad_img)
    f_k = np.fft.fft2(pad_filt)
    inv_fft = np.fft.ifft2(f_i * f_k).real

    skip_h = filt_h // 2
    skip_w = filt_w // 2
    return inv_fft[skip_h:skip_h + img_h, skip_w:skip_w + img_w]



"""
   FFT Filtering (5 Points)

   Apply an ideal low-pass or high-pass filter to an image using the Fourier transform.

   Low frequencies correspond to slowly varying image content, while high 
   frequencies correspond to rapid changes such as edges.

   You are provided with a helper function `radial_frequency_grid(H, W)` that
   returns, for each frequency coefficient in the *shifted* Fourier spectrum,
   its distance from the zero-frequency component.

   Arguments:
      image     - a 2D numpy array (grayscale image)
      cutoff    - positive scalar specifying the cutoff radius in frequency bins
      pass_type - 'low' for low-pass filtering, or 'high' for high-pass filtering

   Returns:
      result    - a 2D numpy array of the same shape as image, containing the
                  filtered image
"""
def radial_frequency_grid(H, W):
    """
    Returns an (H, W) array where each entry is the distance (in frequency bins)
    from the zero-frequency component (DC), assuming fftshifted spectrum.
    """
    cy = H // 2
    cx = W // 2
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    yy, xx = np.meshgrid(y, x, indexing='ij')
    return np.sqrt(xx**2 + yy**2)

def fft_filter(image, cutoff, pass_type='low'):
    ##########################################################################
    h, w = image.shape 

    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)

    frequencies = radial_frequency_grid(h, w)

    if pass_type == 'low': 
       filt = (frequencies <= cutoff)
    elif pass_type == 'high':
       filt = (frequencies >= cutoff)
    fixed = fft_shift * filt

    revert = np.fft.ifftshift(fixed)
    inv_ff_pure = np.fft.ifft2(revert)
    result = np.real(inv_ff_pure)
   
    ##########################################################################
    return result

"""
   SMOOTHING AND DOWNSAMPLING (5 Points)

   Smooth an image by applying a gaussian filter, followed by downsampling with a factor k.

   Note:
      Image downsampling is generally implemented as two-step process:

        (1) Smooth images with low pass filter, e.g, gaussian filter, to remove
            the high frequency signal that would otherwise causes aliasing in
            the downsampled outcome.

        (2) Downsample smoothed images by keeping every kth samples.

      Make an appropriate choice of sigma to avoid insufficient or over smoothing.

      In principle, the sigma in gaussian filter should respect the cut-off frequency
      1 / (2 * k) with k being the downsample factor and the cut-off frequency of
      gaussian filter is 1 / (2 * pi * sigma).


   Arguments:
     image - a 2D numpy array
     downsample_factor - an integer specifying downsample rate

   Returns:
     result - downsampled image, a 2D numpy array with spatial dimension reduced
"""
def smooth_and_downsample(image, downsample_factor = 2):
    ##########################################################################
    sigma = downsample_factor / np.pi 
    smooth = denoise_gaussian(image, sigma)
    result = smooth[::downsample_factor, ::downsample_factor]
    ##########################################################################
    return result

"""
   BILINEAR UPSAMPLING (5 Points)

   Upsampling the input images with a factor of k with bilinear kernel

   Note:
      Image upsampling is generally implemented by mapping each output pixel
      (x_out,y_out) onto input images coordinates (x, y) = (x_out / k, y_out / k).
      Then, we use bilinear kernel to compute interpolated color at pixel
      (x,y), which requires to round (x, y) to find 4 neighboured pixels:

          P11 = (x1, y1)      P12 = (x1, y2)
          P21 = (x2, y1)      P22 = (x2, y2)

      where
          x1 = floor(x / k),  y1 = floor(y / k),
          x2 = ceil (x / k),  y2 = ceil (y / k)

      In practice, you can simplify the 2d interpolation formula by applying 1d
      interpolation along each axis:

          # interpolate along x axis
          C(x,y1) = (x2 - x)/(x2 - x1) * C(x1, y1) +  (x - x1)/(x2 - x1) * C(x2, y1)
          C(x,y2) = (x2 - x)/(x2 - x1) * C(x1, y2) +  (x - x1)/(x2 - x1) * C(x2, y2)

          # interpolate along y axis
          C(x,y) =  (y2 - y)/(y2 - y1) * C(x, y1)  +  (y - y1)/(y2 - y1) * C(x, y2)

      where C(x,y) denotes the pixel color at (x,y).

   Arguments:
     image - a 2D numpy array
     upsample_factor - an integer specifying upsample rate

   Returns:
     result - upsampled image, a 2D numpy array with spatial dimension increased
"""
def bilinear_upsampling(image, upsample_factor = 2):
    ##########################################################################
    height, width = image.shape 
    h_out = height * upsample_factor 
    w_out = width * upsample_factor 

    result = np.zeros((h_out, w_out))

    for h in range(h_out):
       for w in range(w_out):
          y = h / upsample_factor # finding the original coordinates
          x = w / upsample_factor 

          x1 = int(np.floor(x)) # finding the 4 neighboring pixels
          y1 = int(np.floor(y))
          x2 = int(np.ceil(x))
          y2 = int(np.ceil(y))

          x1 = min(max(x1, 0), width - 1) # clamping to image boundaries
          x2 = min(max(x2, 0), width - 1) 
          y1 = min(max(y1, 0), height - 1)
          y2 = min(max(y2, 0), height - 1)
         
          if x2 == x1: # avoid division by zero
               C_x_y1 = image[y1][x1]
               C_x_y2 = image[y2][x1]
          else:
               C_x_y1 = (x2 - x)/(x2 - x1) * image[y1][x1] +  (x - x1)/(x2 - x1) * image[y1][x2]
               C_x_y2 = (x2 - x)/(x2 - x1) * image[y2][x1] +  (x - x1)/(x2 - x1) * image[y2][x2]
          if y2 == y1: # avoid division by zero
               C_x_y = result[h][w] = C_x_y1
          else:
               C_x_y = (y2 - y)/(y2 - y1) * C_x_y1  +  (y - y1)/(y2 - y1) * C_x_y2
               
          result[h][w] = C_x_y
   
    ##########################################################################
    return result

"""
   SOBEL GRADIENT OPERATOR (5 Points)
   Compute an estimate of the horizontal and vertical gradients of an image
   by applying the Sobel operator.
   The Sobel operator estimates gradients dx(horizontal), dy(vertical), of
   an image I as:

         [ 1  0  -1 ]
   dx =  [ 2  0  -2 ] (*) I
         [ 1  0  -1 ]

         [  1  2  1 ]
   dy =  [  0  0  0 ] (*) I
         [ -1 -2 -1 ]

   where (*) denotes convolution.
   Note:
      (1) Your implementation should be as efficient as possible.
      (2) Avoid creating artifacts along the border of the image.
   Arguments:
      image - a 2D numpy array
   Returns:
      dx    - gradient in x-direction at each point
              (a 2D numpy array, the same shape as the input image)
      dy    - gradient in y-direction at each point
              (a 2D numpy array, the same shape as the input image)
"""
def sobel_gradients(image):
   ##########################################################################
   mx = np.array([
      [1, 0, -1], 
      [2, 0, -2], 
      [1, 0, -1]]
   )
   my = np.array(
      [[1, 2, 1], 
      [0, 0, 0], 
      [-1, -2, -1]]
   )
   dx = conv_2d(image, mx)
   dy = conv_2d(image, my)

   ##########################################################################
   return dx, dy

"""
   NONMAXIMUM SUPPRESSION (10 Points)

   Nonmaximum suppression.

   Given an estimate of edge strength (mag) and direction (theta) at each
   pixel, suppress edge responses that are not a local maximum along the
   direction perpendicular to the edge.

   Equivalently stated, the input edge magnitude (mag) represents an edge map
   that is thick (strong response in the vicinity of an edge).  We want a
   thinned edge map as output, in which edges are only 1 pixel wide.  This is
   accomplished by suppressing (setting to 0) the strength of any pixel that
   is not a local maximum.

   Note that the local maximum check for location (x,y) should be performed
   not in a patch surrounding (x,y), but along a line through (x,y)
   perpendicular to the direction of the edge at (x,y).

   A simple, and sufficient strategy is to check if:
      ((mag[x,y] > mag[x + ox, y + oy]) and (mag[x,y] >= mag[x - ox, y - oy]))
   or
      ((mag[x,y] >= mag[x + ox, y + oy]) and (mag[x,y] > mag[x - ox, y - oy]))
   where:
      (ox, oy) is an offset vector to the neighboring pixel in the direction
      perpendicular to edge direction at location (x, y)

   Arguments:
      mag    - a 2D numpy array, containing edge strength (magnitude)
      theta  - a 2D numpy array, containing edge direction in [0, 2*pi)

   Returns:
      nonmax - a 2D numpy array, containing edge strength (magnitude), where
               pixels that are not a local maximum of strength along an
               edge have been suppressed (assigned a strength of zero)
"""
def nonmax_suppress(mag, theta):
   ##########################################################################
   h, w = mag.shape
   nonmax = np.zeros((h, w))

   ang = theta % np.pi
   for x in range(1, h - 1): 
      for y in range(1, w - 1):
          angle = ang[x, y]
          if angle < np.pi / 8 or angle >= 7 * np.pi / 8:
             ox = 0
             oy = 1
          elif np.pi / 8 <= angle < 3 * np.pi / 8:
             ox = 1
             oy = 1
          elif 3 * np.pi / 8 <= angle < 5 * np.pi / 8: 
             ox = 1 
             oy = 0
          else:
             ox = 1
             oy = -1
          m = mag[x, y]
          m_pos = mag[x + ox, y + oy]
          m_neg = mag[x - ox, y - oy]

          if (m > m_pos and m >= m_neg) or (m >= m_pos and m > m_neg):
             nonmax[x, y] = m      

   ##########################################################################
   return nonmax


"""
   HYSTERESIS EDGE LINKING (10 Points)

   Hysteresis edge linking.

   Given an edge magnitude map (mag) which is thinned by nonmaximum suppression,
   first compute the low threshold and high threshold so that any pixel below
   low threshold will be thrown away, and any pixel above high threshold is
   a strong edge and will be preserved in the final edge map.  The pixels that
   fall in-between are considered as weak edges.  We then add weak edges to
   true edges if they connect to a strong edge along the gradient direction.

   Since the thresholds are highly dependent on the statistics of the edge
   magnitude distribution, we recommend to consider features like maximum edge
   magnitude or the edge magnitude histogram in order to compute the high
   threshold.  Heuristically, once the high threshod is fixed, you may set the
   low threshold to be propotional to the high threshold.

   Note that the thresholds critically determine the quality of the final edges.
   You need to carefully tuned your threshold strategy to get decent
   performance on real images.

   For the edge  linking, the weak edges caused by true edges will connect up
   with a neighbouring strong edge pixel.  To track theses edges, we
   investigate the 8 neighbours of strong edges.  Once we find the weak edges,
   located along strong edges' gradient direction, we will mark them as strong
   edges.  You can adopt the same gradient checking strategy used in nonmaximum
   suppression.  This process repeats util we check all strong edges.

   In practice, we use a queue to implement edge linking.  In python, we could
   use a list and its fuction .append or .pop to enqueue or dequeue.

   Arguments:
     nonmax - a 2D numpy array, containing edge strength (magnitude) which is thined by nonmaximum suppression
     theta  - a 2D numpy array, containing edeg direction in [0, 2*pi)

   Returns:
     edge   - a 2D numpy array, containing edges map where the edge pixel is 1 and 0 otherwise.
"""

def hysteresis_edge_linking(nonmax, theta):
   ##########################################################################
   h, w = nonmax.shape 
   edge = np.zeros((h, w)) 

   vals = nonmax[nonmax > 0] 
   if len(vals) == 0: return edge 

   sort_vals = np.sort(vals)
   index_high_pecentile = int(0.9 * (len(sort_vals)-1))
   high = sort_vals[index_high_pecentile]
   low = 0.6 * high 

   strong = nonmax >= high     # mapping the strong edges
   weak = (nonmax >= low) & (nonmax < high ) #mapping the weak edges 


   
   edge[strong] = 1

   angle = theta % np.pi

   # keeps track of idrectionality
   line_direction = np.zeros((h, w))
   line_direction[(angle >= np.pi / 8) & (angle < 3 * np.pi / 8)] = 1 
   line_direction[(angle >= 3 * np.pi / 8) & (angle < 5 * np.pi / 8)] = 2
   line_direction[(angle >= 5 * np.pi / 8) & (angle < 7 * np.pi / 8)] = 3

   neighbor_offsets = [[(0, -1), (0, 1)], [(-1, 1), (1, -1)], [(-1, 0), (1, 0)], [(-1, -1), (1, 1)]]
   

   q = []
   for y in range(h):
      for x in range(w):
         if edge[y, x] == 1:
            q.append((y, x))
   while q: 
      y, x = q.pop() 
      dir = int(line_direction[y, x]) 

      for y_dir, x_dir in neighbor_offsets[dir]: # for each neighbor of the strong edge
         cord_y, cord_x = y + y_dir, x + x_dir 

         if 0 <= cord_y < h and 0 <= cord_x < w:
            if weak[cord_y, cord_x] and edge[cord_y, cord_x] == 0: 
               edge[cord_y, cord_x] = 1
               q.append((cord_y, cord_x))

   ##########################################################################
   return edge


"""
   CANNY EDGE DETECTOR (5 Points)

   Canny edge detector.

   Given an input image:

   (1) Compute gradients in x- and y-directions at every location using the
       Sobel operator.  See sobel_gradients() above.

   (2) Estimate edge strength (gradient magnitude) and direction.

   (3) Run (1)(2) on downsampled images with multiple factors and
       then combine the results via upsampling to original resolution.

   (4) Perform nonmaximum suppression of the edge strength map, thinning it. 
       in the direction perpendicular to that of a local edge.
       See nonmax_suppress() above.

   (5) Compute the high threshold and low threshold of edge strength map
       to classify the pixels as strong edges, weak edges and non edges.
       Then link weak edges to strong edges

   Return the original edge strength estimate (max), the edge
   strength map after nonmaximum suppression (nonmax) and the edge map
   after edge linking (edge)

   Arguments:
      image             - a 2D numpy array
      downsample_factor - a list of interger

   Returns:
      mag      - a 2D numpy array, same shape as input, edge strength at each pixel
      nonmax   - a 2D numpy array, same shape as input, edge strength after nonmaximum suppression
      edge     - a 2D numpy array, same shape as input, edges map where edge pixel is 1 and 0 otherwise.
"""
def canny(image, downsample_factor = [1]):
   ##########################################################################
   h, w = image.shape 

   mag_sum = np.zeros((h, w))
   theta_sum_x = np.zeros((h, w))
   theta_sum_y = np.zeros((h, w))

   for sample in downsample_factor:
      if sample == 1: 
         img_down = image 
      else: 
         img_down = smooth_and_downsample(image, sample)
      
      dx, dy = sobel_gradients(img_down)
      if sample != 1: 
         dx = bilinear_upsampling(dx, sample)
         dy = bilinear_upsampling(dy, sample)

      mag_sample = np.sqrt(dx * dx + dy * dy ) # length of the gradient vector
      theta_sample = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi) # angle of gradient

      mag_sum += mag_sample
      theta_sum_x += np.cos(theta_sample) # circular averaging 
      theta_sum_y += np.sin(theta_sample)
   
   mag = mag_sum / len(downsample_factor)
   theta = np.arctan2(theta_sum_y, theta_sum_x) % (2 * np.pi)
   nonmax = nonmax_suppress(mag, theta)
   edge = hysteresis_edge_linking(nonmax, theta)

   ##########################################################################
   return mag, nonmax, edge
