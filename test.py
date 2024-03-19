from core_of_my_own_square_scripting import *


def output_to_file(output_file, spect_to_sample, batch_x, batch_y, my_y_from_nn, my_cost_per_batch, x_std, x_mean,
                   y_std,
                   y_mean, spect_type="pred"):
    if spect_type not in ["test", "compare", "val", "pred"]:
        raise ValueError("Invalid file_type. Supported values are 'test', 'compare', and 'val'.")
    spect_prefix = {
        "test": "test_out_file_",
        "compare": "compare_test_out_file_",
        "val": "val_test_out_file_",
        "pred": "val_test_out_file_"
    }
    filename = output_file + "/" + spect_prefix[spect_type] + str(spect_to_sample) + ".txt"

    f = open(filename, 'w')
    f.write("XValue\nPredicted\n")
    xvals = batch_x[0] * x_std + x_mean
    for i in list(xvals):
        f.write(str(i) + ",")
    f.write("\n")
    for item in list(my_y_from_nn[0] * y_std + y_mean):
        f.write(str(item) + ",")
    f.flush()
    f.write("\n")
    f.flush()
    f.close()
    print("Cost: ", my_cost_per_batch)
    print(my_y_from_nn)
    print("Wrote to: " + str(filename))


def reading_data_foward(data_file):
    x_file = data_file + 'geometry_for_square_prediction.csv'  # file that contain parameter combination you want to try

    x_data = np.genfromtxt(x_file, delimiter=',', usecols=[0, 1, 2, 3, 4, 5], skip_header=1)
    print(x_data.shape)

    # print("First few lines of x_data and y_data:")
    # for i in range(2):
    # print("x_data[{}]: {}".format(i, x_data[i]))
    # print("y_data[{}]: {}".format(i, y_data[i]))
    # x_mean = x_data.mean(axis=0)
    # x_std = x_data.std(axis=0)

    return x_data


def foward_pred_net(data_file, reuse_weights, output_file, weight_from_model,
                    weight_name_file_to_nn,
                    no_per_batch, numepochs,
                    lr_rate, lr_decay, num_layers, hidden_neurons, percent_val, project_prefix, compare=True,
                    sample_val=True,
                    geometry_to_sample=40, patiencelimit=10):
    x_data = reading_data_foward(data_file)
    x_size = x_data.shape[1]
    tf.compat.v1.disable_eager_execution()  # with tensorflow 2.x eager execution is enabled by default.
    # This allows tf doce to be executed and evaluated line by line.
    with tf.name_scope('input'):
        x = tf.compat.v1.placeholder("float", shape=[None, x_size])
    y_all_std = [3.05586012, 2.16453053, 2.30578243, 1.82490299, 1.7231122, 1.82830763,
                 1.44005349, 1.45755815, 1.28867669, 1.32573753, 1.19073933, 1.20593868,
                 1.20765261, 1.07382398, 1.1261759, 1.06778811, 1.04076806, 1.05278597,
                 1.01571428, 1.11033543, 0.94909731, 1.14476126, 1.04447849, 1.07551952,
                 1.11416608, 1.11179638, 1.08833859, 1.16841828, 1.17354091, 1.19685762,
                 1.29161287, 1.29743295, 1.39109644, 1.46603359, 1.55700618, 1.72525607,
                 1.92403858, 2.25670577, 3.1387965, 3.68385453, 7.5393061, 6.03741974,
                 5.60377988, 4.71094395, 6.77012481, 7.45505218, 7.76826798, 9.2762316,
                 9.67718493, 7.67148735, 7.98381239, 8.75405134, 8.68321, 6.32491947,
                 6.18389824, 6.6988384, 6.38603687, 6.05182718, 6.47350618, 7.25180694,
                 7.91319564, 7.89823722, 8.02909237, 8.38609593, 8.11780336, 8.10406762,
                 7.98096315, 7.53829195, 7.79725227, 7.7609427, 7.21944653, 7.34440789,
                 6.96053348, 6.88685807, 6.39461203, 6.95843135, 6.8626142, 7.13597784,
                 7.36737518, 7.34266798, 7.48631362, 7.53125933, 7.39662652, 7.54174176,
                 7.25324909, 7.50308743, 7.15307118, 7.38023469, 7.03611232, 7.10178261,
                 6.87226313, 6.71547518, 6.58095247, 6.36961039, 6.36435289, 6.14764403,
                 6.27792885, 6.00882505, 6.30482639, 5.97976085, 6.27402187, 6.17479624,
                 6.12744789, 6.5040492, 6.14692043, 6.52422394, 6.41939758, 6.30116906,
                 6.69098984, 6.3140272, 6.43705119, 6.61250476, 6.24624281, 6.43981992,
                 6.43545134, 6.15257778, 6.29682962, 6.25819561, 6.04971988, 6.09592653,
                 6.0599394, 5.9132919, 5.8962295, 5.80984582, 5.71791482, 5.71033219,
                 5.53189227, 5.45680395, 5.52714442, 5.39696201, 5.26460208, 5.33830515,
                 5.35687778, 5.15376051, 5.14908827, 5.28609598, 5.19466006, 5.04756618,
                 5.1285085, 5.24295649, 5.1089464, 5.00900304, 5.10703752, 5.16688505,
                 5.01674358, 4.91876659, 5.00729249, 5.09369908, 4.99144911, 4.88654556,
                 4.92427713, 5.00109692, 4.92707569, 4.79611406, 4.76892501, 4.82673981,
                 4.80950698, 4.68789952, 4.59379969, 4.5993049, 4.62600467, 4.5646941,
                 4.45398724, 4.37481016, 4.35714933, 4.33232862, 4.23964431, 4.1241477,
                 4.03617359, 4.00540726, 3.97339558, 3.91296211, 3.81803656, 3.74141319,
                 3.68450853, 3.63607059, 3.56617337, 3.48051119, 3.38500894, 3.32707542,
                 3.26471319, 3.21146878, 3.14681452, 3.11170283, 3.0769541, 3.06917439,
                 3.04494548, 3.02701491, 2.98652086, 2.931057, 2.86706217, 2.76894152,
                 2.65985497, 2.54745482, 2.46103956, 2.38310116, 2.30866934, 2.22809181,
                 2.13915873, 2.06016729, 1.99419872]
    y_all_mean = [-55.66447559, -54.07797353, -53.49165102, -52.62393975, -52.16020325,
                  -51.16217911, -50.46963085, - 50.08107725, - 49.09836259, - 48.46291957,
                  - 48.12505296, -47.19438634, - 46.66090289, - 46.1369271, - 45.53804346,
                  - 44.74009525, -44.49343868, - 43.67349907, - 43.19386867, - 42.64828208,
                  - 42.09829986, -41.50864046, - 40.98675189, - 40.45004309, - 39.88432654,
                  - 39.30543151, -38.81166554, - 38.21564968, - 37.65828294, - 37.08747627,
                  - 36.56517335, -35.88763583, - 35.30846969, - 34.75284145, - 33.92303854,
                  - 33.34155138, -32.49423345, - 31.5806106, - 30.34982365, - 29.26250397,
                  - 30.97176587, -30.83950756, - 28.98009521, - 28.14306849, - 28.95044405,
                  - 29.1253409, -29.53977051, - 31.00296593, - 31.47080271, - 30.86132127,
                  - 31.52621788, -31.98481063, - 31.41507204, - 29.60978, - 28.78436456,
                  - 28.23623109, -27.56117159, - 27.08229924, - 27.05749291, - 27.45643461,
                  - 28.01043753, -28.51063254, - 29.17719944, - 29.97315813, - 30.45606837,
                  - 30.83965536, -30.90180426, - 30.57743053, - 30.36300236, - 29.93806485,
                  - 29.28571297, -28.88539086, - 28.38092576, - 28.1043843, - 27.79601055,
                  - 27.97040891, -28.0555224, - 28.35437053, - 28.7180605, - 29.03024698,
                  - 29.39784939, -29.71409356, - 29.90725176, - 30.13819781, - 30.12793038,
                  - 30.24889124, -30.06380311, - 30.03274743, - 29.73894647, - 29.5594414,
                  - 29.25819149, -28.98623877, - 28.73576743, - 28.4948858, - 28.34818603,
                  - 28.18211266, -28.15518648, - 28.04717174, - 28.15189493, - 28.09302781,
                  - 28.25975936, -28.32515696, - 28.42888287, - 28.68297808, - 28.70843553,
                  - 28.98214022, -29.08875659, - 29.18902806, - 29.4544751, - 29.43827318,
                  - 29.58496403, -29.72936248, - 29.66573247, - 29.78225012, - 29.8080562,
                  - 29.72115475, -29.77013092, - 29.74109407, - 29.64320809, - 29.62383499,
                  - 29.56913542, -29.47542845, - 29.41854482, - 29.34028662, - 29.26357266,
                  - 29.21151524, -29.11339645, - 29.0561267, - 29.0474163, - 28.98538825,
                  - 28.9324981, -28.94744491, - 28.9489295, - 28.89493172, - 28.90538054,
                  - 28.96055848, -28.95146422, - 28.93480792, - 28.9840105, - 29.04465351,
                  - 29.03466408, -29.03339219, - 29.09116955, - 29.13426282, - 29.11529546,
                  - 29.10817691, -29.15989305, - 29.20387838, - 29.18794507, - 29.16930581,
                  - 29.19006052, -29.21269242, - 29.18771147, - 29.14067344, - 29.12233662,
                  - 29.11890369, -29.09346321, - 29.03051452, - 28.97185791, - 28.93484135,
                  - 28.89293983, -28.82469018, - 28.73732628, - 28.66333763, - 28.60398674,
                  - 28.54457911, -28.47120711, - 28.38812363, - 28.31397288, - 28.24487173,
                  - 28.16780765, -28.08708129, - 27.99768961, - 27.9116017, - 27.82689805,
                  - 27.74379714, -27.65358862, - 27.55570273, - 27.46811203, - 27.39482427,
                  - 27.30385212, -27.23307218, - 27.14945303, - 27.08255478, - 27.0055454,
                  - 26.94483264, -26.87261001, - 26.79875837, - 26.72239916, - 26.64357284,
                  - 26.56165166, -26.48940705, - 26.41508939, - 26.34700891, - 26.28152324,
                  - 26.2291414, -26.17119237, - 26.11197803, - 26.06797882, - 26.01115757,
                  - 25.96392594]

    weights, biases = weights_to_nn(output_file, weight_name_file_to_nn, num_layers, project_prefix)
    # Forward propagation
    y_hat = forward_propagation(x, weights, biases, num_layers)

    start_time = time.time()
    with tf.compat.v1.Session() as sess:
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        print("========                         Iterations started                  ========")
        # Reshape desired_x to have two dimensions

        y_hat_pred = sess.run([y_hat], feed_dict={x: x_data})
        print("y_hat_pred : ", (y_hat_pred[0][0] * y_all_std + y_all_mean))

    print("========Iterations completed in : " + str(time.time() - start_time) + " ========")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    parser.add_argument("--data_file", type=str, default='data/')
    parser.add_argument("--reuse_weights", type=str, default='True')
    parser.add_argument("--output_file", type=str, default='trained_weights/')
    parser.add_argument("--project_prefix", type=str, default="square_")
    parser.add_argument("--weight_name_file_to_nn", type=str, default="")
    parser.add_argument("--weight_from_model", type=str, default="")
    parser.add_argument("--no_per_batch", type=int, default=10)
    parser.add_argument("--numepochs", type=int, default=10000)
    parser.add_argument("--lr_rate", default=.000006)
    parser.add_argument("--lr_decay", default=.9)
    parser.add_argument("--num_layers", default=6)
    parser.add_argument("--hidden_neurons", nargs='+', type=int, default=[6, 60, 180,
                                                                          180, 200, 200])
    parser.add_argument("--num_decay", default=43200)
    parser.add_argument("--percent_val", default=.2)

    args = parser.parse_args()
    dict = vars(args)

    for i in dict:
        if dict[i] == "False":
            dict[i] = False
        elif dict[i] == "True":
            dict[i] = True

    kwargs = {
        'data_file': dict['data_file'],
        'output_file': dict['output_file'],
        'weight_from_model': dict['weight_from_model'],
        'weight_name_file_to_nn': dict['weight_name_file_to_nn'],
        'no_per_batch': dict['no_per_batch'],
        'numepochs': dict['numepochs'],
        'lr_rate': dict['lr_rate'],
        'lr_decay': dict['lr_decay'],
        'num_layers': dict['num_layers'],
        'hidden_neurons': dict['hidden_neurons'],
        'percent_val': dict['percent_val'],
        'reuse_weights': dict['reuse_weights'],
        'project_prefix': dict['project_prefix']
    }

    foward_pred_net(**kwargs)
