import numpy as np
import math
import matplotlib.pyplot as plt
import time


def transformation(reading):
    coordinates = []
    for i in range(len(reading)):
        theta = i * np.pi / 180
        r = reading[i]
        if r != 0:
            coordinates.append([r * math.cos(theta), r * math.sin(theta), 1])
    coordinates = np.array(coordinates)

    # Translation + Rotation
    theta = np.deg2rad(10)
    R = np.array([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    t_x = 2
    t_y = -1
    T = np.array([[1, 0, t_x], [0, 1, t_y], [0, 0, 1]])
    transformed_coordinates = [np.dot(np.dot(R, T), point) for point in coordinates]

    return coordinates, transformed_coordinates

def noise(transformed_coordinates, std):
    noisy_coordinates = []
    for point in transformed_coordinates:
        noisy_coordinates.append([np.random.normal(point[0], std), np.random.normal(point[1], std), 1])
    noisy_coordinates = np.array(noisy_coordinates)

    return noisy_coordinates

# ICP
def icp(coordinates, noisy_coordinates):
    mc_coordinates = np.mean(coordinates, axis=0)
    mc_noisy_coordinates = np.mean(noisy_coordinates, axis=0)

    translation_vec = mc_noisy_coordinates - mc_coordinates
    noisy_coordinates = noisy_coordinates - translation_vec

    error = 1e12
    last_error = np.inf
    iteration = 0
    sol_final = np.eye(3)
    while error/last_error < 0.95 and iteration < 5:
        last_error = error
        iter_start = time.time()

        # plt.scatter([point[0] for point in coordinates], [point[1] for point in coordinates])
        # plt.scatter([point[0] for point in noisy_coordinates], [point[1] for point in noisy_coordinates])
        # plt.legend(("Original", "Transformed"))
        # plt.pause(0.1)
        # plt.clf()

        nearest_neighbor = np.zeros((len(noisy_coordinates), 3))
        for i, point in enumerate(noisy_coordinates):
            distance = np.inf
            for neighbor in coordinates:
                d = np.sum(np.square(point - neighbor))
                if d < distance:
                    distance = d
                    possnearest_neighbor = neighbor
            nearest_neighbor[i] = possnearest_neighbor

        # print(np.array(noisy_coordinates).shape)
        # print(noisy_coordinates[0])
        # print(np.array(nearest_neighbor).shape)
        sol = np.linalg.lstsq(noisy_coordinates, nearest_neighbor, rcond=None)[0]
        error = np.linalg.norm(noisy_coordinates - nearest_neighbor)
        sol_final = sol_final @ sol

        noisy_coordinates = [np.dot(point, sol) for point in noisy_coordinates]
        iteration += 1
        # print("Iteration time:", time.time() - iter_start)
        # print("Error:", error / last_error)

    return sol_final, translation_vec

if __name__ == "__main__":
    reading = [3.799999952316284, 3.684000015258789, 2.8499999046325684, 2.8380000591278076, 2.8529999256134033, 2.871999979019165, 2.865000009536743, 2.884999990463257, 0.0, 0.0, 0.0, 1.843999981880188, 1.7300000190734863, 1.7000000476837158, 1.656000018119812, 2.065000057220459, 1.562999963760376, 1.5260000228881836, 1.5820000171661377, 1.5679999589920044, 1.5880000591278076, 1.6030000448226929, 1.6150000095367432, 1.6330000162124634, 1.6699999570846558, 1.6740000247955322, 1.6859999895095825, 1.7000000476837158, 1.718000054359436, 1.7369999885559082, 1.7640000581741333, 1.8220000267028809, 1.8200000524520874, 1.840999960899353, 1.8700000047683716, 1.8960000276565552, 1.9270000457763672, 1.9450000524520874, 1.9739999771118164, 2.01200008392334, 2.0360000133514404, 2.0510001182556152, 2.109999895095825, 2.1480000019073486, 2.1510000228881836, 2.2109999656677246, 2.259000062942505, 2.322000026702881, 2.390000104904175, 2.434000015258789, 2.496000051498413, 2.628999948501587, 2.694999933242798, 2.742000102996826, 2.7760000228881836, 2.869999885559082, 2.9609999656677246, 3.0390000343322754, 3.190999984741211, 3.2179999351501465, 3.4260001182556152, 3.7019999027252197, 3.7869999408721924, 3.8929998874664307, 4.182000160217285, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5369999408721924, 2.555999994277954, 2.5169999599456787, 0.0, 0.0, 1.6619999408721924, 1.4110000133514404, 1.3020000457763672, 1.1469999551773071, 1.0470000505447388, 0.9879999756813049, 0.8740000128746033, 0.8700000047683716, 0.8669999837875366, 0.8659999966621399, 0.8679999709129333, 0.8740000128746033, 0.8769999742507935, 0.8899999856948853, 0.8870000243186951, 0.9010000228881836, 0.9089999794960022, 0.9169999957084656, 0.9279999732971191, 0.9350000023841858, 0.9480000138282776, 0.9589999914169312, 0.968999981880188, 0.9769999980926514, 0.9810000061988831, 0.9869999885559082, 0.9990000128746033, 1.034999966621399, 1.034000039100647, 1.0449999570846558, 1.0700000524520874, 1.065000057220459, 1.0779999494552612, 1.0989999771118164, 1.1160000562667847, 1.1089999675750732, 1.1319999694824219, 1.1549999713897705, 1.1670000553131104, 1.1990000009536743, 1.2259999513626099, 1.2380000352859497, 1.2640000581741333, 1.3020000457763672, 1.3309999704360962, 1.3660000562667847, 1.3639999628067017, 1.3910000324249268, 1.434000015258789, 1.4559999704360962, 1.4919999837875366, 1.531999945640564, 1.562999963760376, 1.6100000143051147, 1.6440000534057617, 1.74399995803833, 1.8040000200271606, 1.871000051498413, 1.937000036239624, 2.010999917984009, 2.0739998817443848, 2.1519999504089355, 2.25, 2.3519999980926514, 2.559000015258789, 2.680000066757202, 2.7960000038146973, 2.8510000705718994, 3.0260000228881836, 3.2669999599456787, 3.38100004196167, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.2290000915527344, 3.065000057220459, 2.9800000190734863, 0.0, 2.875, 2.7130000591278076, 2.628000020980835, 2.502000093460083, 2.127000093460083, 2.066999912261963, 2.005000114440918, 1.8919999599456787, 1.8420000076293945, 1.8109999895095825, 1.746000051498413, 1.625, 1.590000033378601, 1.531999945640564, 1.5089999437332153, 1.4759999513626099, 1.4290000200271606, 1.402999997138977, 1.3940000534057617, 1.3619999885559082, 1.3359999656677246, 1.3049999475479126, 1.2730000019073486, 1.253999948501587, 1.2100000381469727, 1.1959999799728394, 1.1679999828338623, 1.149999976158142, 1.1410000324249268, 1.1139999628067017, 1.090999960899353, 1.0800000429153442, 1.065999984741211, 1.0759999752044678, 1.0520000457763672, 1.0290000438690186, 1.0360000133514404, 0.9850000143051147, 0.9850000143051147, 0.9829999804496765, 0.9750000238418579, 0.9760000109672546, 0.9629999995231628, 0.9459999799728394, 0.9380000233650208, 0.9300000071525574, 0.9129999876022339, 0.8980000019073486, 0.8899999856948853, 0.8849999904632568, 0.8769999742507935, 0.8799999952316284, 0.8769999742507935, 0.8640000224113464, 0.8619999885559082, 0.8529999852180481, 0.8519999980926514, 0.8500000238418579, 0.8339999914169312, 0.8410000205039978, 0.8370000123977661, 0.8339999914169312, 0.8320000171661377, 0.8289999961853027, 0.828000009059906, 0.8240000009536743, 0.8309999704360962, 0.8299999833106995, 0.828000009059906, 0.8230000138282776, 0.8270000219345093, 0.8240000009536743, 0.8259999752044678, 0.8199999928474426, 0.8180000185966492, 0.824999988079071, 0.8240000009536743, 0.824999988079071, 0.8240000009536743, 0.8209999799728394, 0.8299999833106995, 0.8289999961853027, 0.8309999704360962, 0.8330000042915344, 0.8389999866485596, 0.8460000157356262, 0.8420000076293945, 0.8519999980926514, 0.8650000095367432, 0.8629999756813049, 0.8600000143051147, 0.8629999756813049, 0.8759999871253967, 0.8849999904632568, 0.8899999856948853, 0.8980000019073486, 0.9039999842643738, 0.9049999713897705, 0.9240000247955322, 0.9359999895095825, 0.9520000219345093, 0.9599999785423279, 0.9700000286102295, 0.9769999980926514, 0.9819999933242798, 0.9909999966621399, 1.0069999694824219, 1.0199999809265137, 1.0579999685287476, 1.055999994277954, 1.0759999752044678, 1.0609999895095825, 1.100000023841858, 1.1109999418258667, 1.1089999675750732, 1.1369999647140503, 1.1399999856948853, 1.1670000553131104, 1.2020000219345093, 1.225000023841858, 1.253000020980835, 1.2680000066757202, 1.3020000457763672, 1.3370000123977661, 1.3739999532699585, 1.3799999952316284, 1.4140000343322754, 1.4450000524520874, 1.4539999961853027, 1.5019999742507935, 1.5470000505447388, 1.5959999561309814, 1.621999979019165, 1.6519999504089355, 1.6339999437332153, 1.621000051498413, 1.5950000286102295, 1.565999984741211, 1.5290000438690186, 1.6239999532699585, 1.6339999437332153, 1.6299999952316284, 1.6160000562667847, 1.6039999723434448, 1.5839999914169312, 1.5779999494552612, 1.5789999961853027, 1.5640000104904175, 1.5490000247955322, 1.5399999618530273, 0.0, 0.0, 2.8450000286102295, 2.8459999561309814, 2.9019999504089355, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.874000072479248]

    coordinates, transformed_coordinates = transformation(reading)
    noisy_coordinates = noise(transformed_coordinates, std=0.05)
    sol_final, translation_vec = icp(coordinates, noisy_coordinates)

    result = coordinates @ np.linalg.inv(sol_final) + translation_vec

    plt.scatter([point[0] for point in transformed_coordinates], [point[1] for point in transformed_coordinates])
    plt.scatter([point[0] for point in result], [point[1] for point in result])
    plt.legend(("Original", "Transformed"))
    plt.show()