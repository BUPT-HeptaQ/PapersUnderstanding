
Step 1: Gaussian Blur and get 5-6 ambiguity pictures. L(x, y, σ)  = G(x, y, σ)  * I(x, y) 
and Gaussian formulation is G(x, y, σ) = 1/(2pi*σ^2) * e^-(x^2+y^2)/2σ^2. The larger σ, the higher the degree of Gaussian blur

Step 2: Constructing multi-resolution pyramids directly using downsampling without blurring, 
where average downsampling can be used. direct downsampling to obtain multiresolution images

Step 3: A Gaussian difference pyramid is constructed. Each group of five original pictures are the pictures blurred by different σ Gaussian parameters. 
Perform 5 subtraction operations on the top and bottom of the graph to obtain a difference map. Using the DOG formulation:
D(x, y, σ) = [G(x, y, kσ) – G(x, y, σ)] * I(x, y) = L(x, y, kσ) – L(x, y, σ)
This equation represents a Gaussian difference pyramid, that is, different Gaussian terms are subtracted, 
and finally *I(x, y) represents the size of the differential pyramid.

Step 4: For the Gaussian difference pyramid obtained, find the extreme point. For a point is an extreme point or not, the 9 points corresponding to the upper picture plus the 9 points corresponding to the next picture, plus 8 points around this point to determine whether this point is an extreme point

Step 5: If it is an extreme point, it is the key point. 
Here we make a precise positioning of the key points and use the Taylor formula to expand:
D(x) = D + ∂ D^T / ∂x * x + 1/2 * x^T * ∂D^2 / ∂^X^2 * x 
x represents the offset on the x-axis, for x the derivation is equal to 0, 
and final result is returned to D(x), where D(x) is the final extreme point.

Step 6: Use the herrian formula to compare the magnitudes of λ1 and λ2 by eigenvector changes to eliminate boundary Eliminate boundary effects
Using the principle of harris corner detection, find H(x, y) is the gradient variation matrix of the structure, 
and solve for λ1 and λ2. If λ1>>λ2 is expressed as the boundary point, remove it. 
H(x,y)=[■(D_xx (x,y)&D_xy (x,y)@D_xy (x,y)&D_yy (x,y))]
Tr(H)=D_xx+D_yy= α+ β
Det(H)=D_xx D_yy-(D_xy )^2= αβ
〖Tr(H)〗^2/(Det(H))=〖(α+ β)〗^2/αβ=〖(γ+1)〗^2/γ
The experiment in this paper use γ=10, which eliminates keypoints that have a ratio between the principle curvatures greater than 10. 
And let α be the eigenvalue with the largest magnitude and β be the smaller one.

Step 7: Using the sobel operator, each feature point gets three pieces of information, gets the position, 
calculates the size of the gradient, and the direction of the gradient.
For each image sample, L(x, y), at this scale, the gradient magnitude, m(x, y), and orientation, 
θ(x, y), is precomputed using pixel differences:
m(x, y) = [(L(x+1), y – L(x-1), y)^2 + (L(x, y+1) – L(x, y-1))^2]^1/2
θ(x, y) = tan^(-1) ((L(x, y+1) – L(x, y-1))/ (L(x+1), y – L(x-1), y)))

Step 8: Count the direction of the gradient of the adjacent part, draw a histogram, 
and take the most occurrences in the histogram as the main direction. 
If the number of times in the secondary direction is greater than 0.8 in the main direction, 
then the secondary direction is also the auxiliary direction.

Step 9: corresponding to the main direction of the feature, rotating, maintaining the direction invariance of the feature point size
Step 10: Statistics on the domain feature points. According to the number of 4*4, 
each region generates 8 directions, that is, the number of occurrences in each direction is used as a feature.
Generally, 16 regions are used, that is, 16*8=128. Features
Shift feature point: used to detect and describe the characteristics of a picture, 
it finds extreme points in spatial scale, and extracts position, scale (gradient size), rotation invariant (direction)


