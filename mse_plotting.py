import image_methods
import numpy as np
import matplotlib.pyplot as plt

# 8x8
# b_mse = [916.4079373554408, 445.56384162925366,
#          224.4894271944275, 113.07215889662434,
#          38.97056462540206]
# a_mse = [748.7113171254011, 411.1653115140807,
#          260.79496854050194, 181.54635329199553, 139.37475330643235]
# s = [10, 20, 30, 40, 50]
#
# plt.plot(s, b_mse, label="Without Median Filtering", marker='*', linestyle='--')
# plt.plot(s, a_mse, label="With Median Filtering", marker='*', linestyle='--')
# plt.xlabel("Number of Samples Per Block")
# plt.ylabel("MSE between Recovered Image and Original Image")
# plt.title("Recovery Error vs. Number of Samples with and without Median Filtering")
# plt.xticks(np.arange(0, 55, 10))
# plt.legend()
# plt.show()

# 4x4
b_mse = [1597.6239471924457, 924.5953545765028,
         398.48118523157393, 237.29412024199365,
         142.66462884323565]
a_mse = [933.0988475679344, 523.9406380209657,
         337.73179579185734, 249.93651480769478,
         191.87063111452923]
s = [2, 4, 6, 8, 10]
plt.plot(s, b_mse, label="Without Median Filtering", marker='*', linestyle='--')
plt.plot(s, a_mse, label="With Median Filtering", marker='*', linestyle='--')
plt.xlabel("Number of Samples Per Block")
plt.ylabel("MSE between Recovered Image and Original Image")
plt.xticks(s)
plt.title("Recovery Error vs. Number of Samples with and without Median Filtering")
plt.legend()
plt.show()

# 16x16
# b_mse = [887.3808371920459, 561.955590245396, 443.37776805460163, 264.768092421538, 154.4752317836555]
# a_mse = [774.9180608716736, 516.502846918607, 436.7105080345377, 333.5478451821609, 273.20743996965734]
# s = [10, 30, 50, 100, 150]
# plt.plot(s, b_mse, label="Without Median Filtering", marker='*', linestyle='--')
# plt.plot(s, a_mse, label="With Median Filtering", marker='*', linestyle='--')
# plt.xlabel("Number of Samples Per Block")
# plt.ylabel("MSE between Recovered Image and Original Image")
# plt.xticks(s)
# plt.title("Recovery Error vs. Number of Samples with and without Median Filtering")
# plt.legend()
# plt.show()