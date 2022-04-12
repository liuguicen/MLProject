import image_util
from common_lib_import_and_set import *
import matplotlib.pyplot as plt
from scipy.stats import norm


def generateTracklet():
    m = 200  # measurements
    vx = 20
    vy = 10

    mx = np.array(vx + np.random.randn(m))
    my = np.array(vy + np.random.randn(m))
    measurements = np.vstack((mx, my))
    print('22342423423423423')
    print(measurements.shape)
    print('Standard Deviation of Acceleration Measurements=%0.2f' % np.std(mx))
    return m, mx, my, measurements


# print('You assumed %0.2f in R.' % norm.R[0, 0])
def showTrackelet(m, mx, my):
    fig = plt.figure(figsize=(16, 5))
    plt.step(range(m), mx, label='$\dot x $')
    plt.step(range(m), my, label='$\dot y $')
    plt.ylabel(r'Velocity $m/s$')
    plt.title('Measurements')
    plt.legend(loc='best', prop={'size': 18})
    plt.savefig('measurements.png')
    image_util.showImage('measurements.png')


def plot_xy(measurements):
    fig = plt.figure(figsize=(16, 9))
    plt.scatter(xt, yt, s=20, label='State', c='k')
    plt.scatter(xt[0], yt[0], s=100, label='Start', c='g')
    plt.scatter(xt[-1], yt[-1], s=100, label='Goal', c='r')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend('Position')
    plt.axis('equal')
    plt.show()

def save_states(x, Z, P, R, K):
    xt.append(float(x[0]))
    yt.append(float(x[1]))
    dxt.append(float(x[2]))
    dyt.append(float(x[3]))

    Zx.append(float(Z[0]))
    Zy.append(float(Z[1]))

    Px.append(float(P[0, 0]))
    Py.append(float(P[1, 1]))
    Pdx.append(float(P[2, 2]))
    Pdy.append(float(P[3, 3]))

    Rdx.append(float(R[0, 0]))
    Rdy.append(float(R[1, 1]))

    Kx.append(float(K[0, 0]))
    Ky.append(float(K[1, 0]))
    Kdx.append(float(K[2, 0]))
    Kdy.append(float(K[3, 0]))


def plot_x(measurements):
    # fig = plt.figure(figsize=(16, 9))
    plt.step(range(len(measurements[0])), dxt, label='$estimateVx $')
    plt.step(range(len(measurements[0])), dyt, label='$estimateVy $')

    plt.step(range(len(measurements[0])), measurements[0], label='$measurementVx$')
    plt.step(range(len(measurements[0])), measurements[1], label='$measurementVy$')

    # plt.axhline(vx, colors='#999999',label = '$trueVx$')
    # plt.axhline(vy, colors='#999999',label = '$trueVy$')

    plt.xlabel('Filter Step')
    plt.title('Estimate (Elements from State Vector $x$)')
    plt.legend(loc='best', prop={'size': 11})
    plt.ylim([0, 30])
    plt.ylabel('Velocity')
    plt.show()


if __name__ == "__main__":
    m, mx, my, measurements = generateTracklet()
    showTrackelet(m, mx, my)
    x = np.matrix([[0.0, 0.0, 0.0, 0.0]]).T
    print('行人初始状态\n', x, x.shape)
    P = np.diag([1000.0, 1000.0, 1000.0, 1000.0])
    print('行人不确定性，协方差矩阵 \n', P, P.shape)
    dt = 0.1  # Time step between Filters steps
    F = np.matrix([[1.0, 0.0, dt, 0.0],
                   [0.0, 1.0, 0.0, dt],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])
    print('状态转移矩阵\n', F, F.shape)

    # ‘‘‘
    # sv = 0.5
    # G = np.matrix([[0.5 * dt ** 2],
    #                [0.5 * dt ** 2],
    #                [dt],
    #                [dt]])
    # Q = G * G.T * sv * 2
    # from sympy import Symbol, Matrix
    # from sympy.interactive import printing
    #
    # printing.init_printing()
    # dts = Symbol('dt')
    # ’’’

    noise_ax = 0.5
    noise_ay = 0.5
    dt_2 = dt * dt
    dt_3 = dt_2 * dt
    dt_4 = dt_3 * dt

    Q = np.matrix([[0.25 * dt_4 * noise_ax, 0, 0.5 * dt_3 * noise_ax, 0],
                   [0, 0.25 * dt_4 * noise_ay, 0, 0.25 * dt_3 * noise_ay],
                   [dt_3 / 2 * noise_ax, 0, dt_2 * noise_ax, 0],
                   [0, dt_3 / 2 * noise_ay, 0, dt_2 * noise_ay]])

    H = np.matrix([[0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])
    print('测量矩阵\n', H, H.shape)
    ra = 0.09  # 厂商提供
    R = np.matrix([[ra, 0.0],
                   [0.0, ra]])
    print('传感器协方差矩阵\n', R, R.shape)

    I = np.eye(4)
    print('单位向量\n', I, I.shape)

    xt = []
    yt = []
    dxt = []
    dyt = []

    Zx = []
    Zy = []

    Px = []
    Py = []
    Pdx = []
    Pdy = []

    Rdx = []
    Rdy = []

    Kx = []
    Ky = []
    Kdx = []
    Kdy = []

    for n in range(len(measurements[0])):
        # Time Update(Prediction)
        # ==============================
        x = F * x  # Project the state ahead
        P = F * P * F.T + Q  # Project the error covariance ahead

        # Measurement Update (Correction)
        # ==============================
        S = H * P * H.T + R
        K = (P * H.T) * np.linalg.pinv(S)

        # Update the estimate via z
        Z = measurements[:, n].reshape(2, 1)
        y = Z - (H * x)
        x = x + (K * y)

        # update the error convariance
        P = (I - (K * H)) * P

        # save states (for Plotting)
        save_states(x, Z, P, R, K)
    plot_x(measurements)
    plot_xy(measurements)
