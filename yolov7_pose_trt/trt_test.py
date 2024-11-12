import torch
import time
import cv2
# a = torch.tensor([324.5209, 313.5145, 134.5660, 243.6360,   0.9485,   0.9835, 331.7599,
#         340.8721, 327.8386, 355.2708, 321.5312, 364.3750, 305.0000, 381.3750,
#         281.4375, 363.0312, 269.8750, 342.5625, 308.0234, 337.9375, 312.1484,
#         334.4688, 312.5859, 226.5688, 221.6748, 218.9004, 226.4786, 218.0938,
#         259.7422, 250.0625, 301.5469, 278.2500, 295.4375, 265.3125, 342.5000,
#         337.7500, 384.2500, 379.8125, 412.5000, 405.0000,   0.9956,   0.9966,
#           0.9795,   0.9756,   0.4141,   0.9868,   0.9849,   0.9683,   0.9453,
#           0.9653,   0.9404,   0.9834,   0.9819,   0.9609,   0.9536,   0.8994,
#           0.8892]).to('cuda:0')


# t1 = time.time()
# b = a.cpu()
# t2 = time.time()
# print("gpu to cpu:", (t2-t1)*1000,"ms")

# bbox = b[0:4]
# t21 = time.time()
# print("1:", (t21-t2)*1000,"ms")

# bbox_score = b[4]
# t22 = time.time()
# print("2:", (t22-t21)*1000,"ms")

# # key_points = torch.zeros(17)
# # index = torch.tensor([6,23,7,24])
# # key_points = b[index.long()]
# c=b.tolist()
# key_list = []
# for i in range(17):
#     # key_list = (c[6],c[23],c[7],c[24],)
#     key_list.append(c[i+6])
#     key_list.append(c[i+23])
# key_points = torch.tensor(key_list).view(17,2)
# t23 = time.time()
# print("3:", (t23-t22)*1000,"ms")

# kp_score = b[40:].view(17,1)
# t24 = time.time()
# print("4:", (t24-t23)*1000,"ms")

# print("extract information:", (t24-t2)*1000,"ms")
# final_result = []
# final_result.append({
#         'bbox': bbox,
#         'bbox_score': bbox_score,
#         'keypoints': key_points,
#         'kp_score': kp_score
#     })

# t3 = time.time()
# print("append final result:", (t3-t24)*1000,"ms")
# print("total process time:", (t3-t1)*1000,"ms")
# print(final_result)

def resizeResults(results, img_shape, res_shape, top_left, right_down):
    #img_shape: 1080*1920
    #res_shape:640*640
    k1 = img_shape[0]/res_shape[0]
    k2 = img_shape[1]/res_shape[1]
    k = max(k1, k2)
    k3 = img_shape[0]/img_shape[1]
    margin = (res_shape[0]-res_shape[1]*k3)/2
    # print(k1," ",k2,"",margin)
    result_list = results.tolist()
    final_result = []
    for res in result_list:
        res[0] = res[0]*k
        res[1] = (res[1]-margin)*k
        res[2] = res[2]*k
        res[3] = res[3]*k
        for i in range(17):
            res[i+6] = res[i+6]*k
            res[i+23] = (res[i+23]-margin)*k
        print((res[21]+res[22])/2, ' ',(res[38]+res[39])/2)
        if ((res[21]+res[22])/2 > top_left[0]) & ((res[21]+res[22])/2 < right_down[0]) \
            & ((res[38]+res[39])/2 > top_left[1]):
            final_result.append(res)
        else:
            continue
    results = torch.tensor(final_result).to('cuda:0')
    return results


a = torch.tensor([[ 3.2452e+02,  3.1351e+02,  1.3457e+02,  2.4364e+02,  9.4854e-01,
          9.8348e-01,  3.3176e+02,  3.4087e+02,  3.2784e+02,  3.5527e+02,
          3.2153e+02,  3.6438e+02,  3.0500e+02,  3.8138e+02,  2.8144e+02,
          3.6303e+02,  2.6988e+02,  3.4256e+02,  3.0802e+02,  3.3794e+02,
          3.1215e+02,  3.3447e+02,  3.1259e+02,  2.2657e+02,  2.2167e+02,
          2.1890e+02,  2.2648e+02,  2.1809e+02,  2.5974e+02,  2.5006e+02,
          3.0155e+02,  2.7825e+02,  2.9544e+02,  2.6531e+02,  3.4250e+02,
          3.3775e+02,  3.8425e+02,  3.7981e+02,  4.1250e+02,  4.0500e+02,
          9.9561e-01,  9.9658e-01,  9.7949e-01,  9.7559e-01,  4.1406e-01,
          9.8682e-01,  9.8486e-01,  9.6826e-01,  9.4531e-01,  9.6533e-01,
          9.4043e-01,  9.8340e-01,  9.8193e-01,  9.6094e-01,  9.5361e-01,
          8.9941e-01,  8.8916e-01],
        [ 2.3308e+01,  3.4396e+02,  4.6608e+01,  1.9445e+02,  6.8210e-01,
          9.8348e-01,  2.0345e+00,  2.2350e+00,  1.2698e+00, -8.7326e-01,
          2.6562e-01, -8.4375e-01,  3.1250e-02,  5.1250e+00,  4.6250e+00,
          2.6125e+01,  2.5656e+01,  7.8125e-02, -3.1250e-02,  4.7812e+00,
          3.2656e+00,  2.6094e+00,  4.1875e+00,  2.8473e+02,  2.8115e+02,
          2.8094e+02,  2.8198e+02,  2.8122e+02,  2.9888e+02,  3.0062e+02,
          3.2056e+02,  3.2375e+02,  3.3931e+02,  3.4112e+02,  3.4888e+02,
          3.5038e+02,  3.6975e+02,  3.7300e+02,  4.1638e+02,  4.2025e+02,
          1.7303e-02,  6.9580e-03,  1.9608e-02,  2.1652e-02,  3.6285e-02,
          2.0239e-01,  5.2612e-02,  4.5508e-01,  1.5332e-01,  6.4941e-01,
          4.2163e-01,  2.5781e-01,  1.2659e-01,  3.3643e-01,  1.9775e-01,
          3.0566e-01,  2.0483e-01]], device='cuda:0')


img_shape = [1080, 1920]
res_shape = [640, 640]
topleftX = 600
topleftY = 600
rightdownX = 1000
rightdownY = 900
t1 = time.time()
results = resizeResults(a, img_shape, res_shape, [topleftX, topleftY], [rightdownX, rightdownY])
t2 = time.time()
print("resize results:", (t2-t1)*1000,"ms")
print(results)
img = cv2.imread('test1.png')
img = cv2.resize(img, (960, 540))
cv2.imshow("img",img)
cv2.waitKey(0)

# t1 = time.time()
# d = a.cpu()
# t2 = time.time()
# print("gpu to cpu:", (t2-t1)*1000,"ms")
# final_result = []
# for b in d:
#     t20 = time.time()
#     temp = b[0:4].tolist()
#     bbox=[]
#     bbox.append(temp[0] - temp[2]/2)
#     bbox.append(temp[1] - temp[3]/2)
#     bbox.append(temp[0] + temp[2]/2)
#     bbox.append(temp[1] + temp[3]/2)
#     bbox=torch.tensor(bbox)
#     t21 = time.time()
#     print("1:", (t21-t20)*1000,"ms")

#     bbox_score = b[4]
#     t22 = time.time()
#     print("2:", (t22-t21)*1000,"ms")

#     # key_points = torch.zeros(17)
#     # index = torch.tensor([6,23,7,24])
#     # key_points = b[index.long()]
#     c=b.tolist()
#     key_list = []
#     for i in range(17):
#         # key_list = (c[6],c[23],c[7],c[24],)
#         key_list.append(c[i+6])
#         key_list.append(c[i+23])
#     key_points = torch.tensor(key_list).view(17,2)
#     t23 = time.time()
#     print("3:", (t23-t22)*1000,"ms")

#     kp_score = b[40:].view(17,1)
#     t24 = time.time()
#     print("4:", (t24-t23)*1000,"ms")

#     print("extract information:", (t24-t2)*1000,"ms")

#     final_result.append({
#             'bbox': bbox,
#             'bbox_score': bbox_score,
#             'keypoints': key_points,
#             'kp_score': kp_score
#         })
#     t3 = time.time()
#     print("append final result:", (t3-t24)*1000,"ms")

# print("total process time:", (time.time()-t1)*1000,"ms")


# print(final_result)










# [{'bbox': tensor([611.,   0., 937., 864.]), 
# 'bbox_score': tensor(0.76570), 
# 'keypoints': tensor([[804.52142, 200.03928],
#         [819.93219, 200.03928],
#         [789.11072, 184.62857],
#         [850.75360, 200.03928],
#         [773.70001, 184.62857],
#         [866.16431, 292.50357],
#         [742.87860, 261.68216],
#         [912.39642, 384.96786],
#         [665.82501, 323.32501],
#         [866.16431, 369.55716],
#         [650.41431, 292.50357],
#         [819.93219, 477.43216],
#         [742.87860, 477.43216],
#         [804.52142, 585.30713],
#         [742.87860, 569.89642],
#         [804.52142, 631.53931],
#         [742.87860, 631.53931]]), 
# 'kp_score': tensor([[0.90344],
#         [0.93574],
#         [0.93320],
#         [0.93215],
#         [0.92355],
#         [0.88916],
#         [0.90810],
#         [0.86611],
#         [0.85610],
#         [0.92378],
#         [0.99222],
#         [0.80739],
#         [0.83065],
#         [0.67463],
#         [0.74502],
#         [0.84232],
#         [0.79878]]), 
# 'proposal_score': tensor([2.87435])}]
