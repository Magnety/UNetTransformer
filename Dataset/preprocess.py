import SimpleITK as sitk
import numpy as np
import os
import cv2
def Origin(name,dir,img_np,label_np,indate):
    dir = dir+'/origin/'+name
    if not os.path.isdir(dir):
        os.makedirs(dir)
    np.save(dir+'/imgae.npy',img_np)
    np.save(dir+'/label.npy',label_np)
    target = open(dir+'/label.txt', 'w')  # 打开目的文件
    target.write(indate)


def Slice(name,dir,img_np,label_np,indate):
    a, b, c = [], [], []
    for i in range(label_np.shape[0]):
        (mean, stddv) = cv2.meanStdDev(label_np[i, :, :])
        if mean > 0:
            a.append(i)
    for i in range(label_np.shape[1]):
        (mean, stddv) = cv2.meanStdDev(label_np[:, i, :])
        if mean > 0:
            b.append(i)
    for i in range(label_np.shape[2]):
        (mean, stddv) = cv2.meanStdDev(label_np[:, :, i])
        if mean > 0:
            c.append(i)
    # abc不正常
    a_wise = label_np.shape[0]
    b_wise = label_np.shape[1]
    c_wise = label_np.shape[2]
    a_s = a[0]
    a_e = a[len(a) - 1]
    b_s = b[0]
    b_e = b[len(b) - 1]
    c_s = c[0]
    c_e = c[len(c) - 1]

    """if (a[len(a)-1]+a[0])//2<32:
        a_s = 0
        a_e = 64
    elif (a[len(a)-1]+a[0])//2>a_wise-32:
        a_s = a_wise-64
        a_e = a_wise
    else:
        center =  (a[len(a)-1]+a[0])//2
        a_s = center-32
        a_e = center+32

    if (b[len(b) - 1] + b[0]) // 2 < 128:
        b_s = 0
        b_e = 256
    elif (b[len(b) - 1] + b[0]) // 2 > b_wise - 128:
        b_s = b_wise - 256
        b_e = b_wise
    else:
        center = (b[len(b) - 1] + b[0]) // 2
        b_s = center - 128
        b_e = center + 128

    if (c[len(c) - 1] + c[0]) // 2 < 128:
        c_s = 0
        c_e = 256
    elif (c[len(c) - 1] + c[0]) // 2 > c_wise - 128:
        c_s = c_wise - 256
        c_e = c_wise
    else:
        center = (c[len(c) - 1] + c[0]) // 2
        c_s = center - 128
        c_e = center + 128"""


    if (a_e-a_s)<=44 or (a_e-a_s)>64:
        if a_s >= 10:
            a_s -= 10
        else:
            a_s = 0
        if a_e <= 308:
            a_e += 10
        else:
            a_e = 318

    if (b_e - b_s) <=236 or (b_e - b_s) > 256:
        if b_s >= 10:
            b_s -= 10
        else:
            b_s = 0
        if b_wise == 443:
            if b_e <= 433:
                b_e += 10
            else:
                b_e = 443
        elif b_wise == 507:
            if b_e <= 497:
                b_e += 10
            else:
                b_e = 507
        else:
            if b_e <= 585:
                b_e += 10
            else:
                b_e = 595

    if (c_e - c_s) <=236 or (c_e - c_s) > 256:
        if c_s >= 10:
            c_s -= 10
        else:
            c_s = 0
        if c_wise == 697:
            if c_e <= 687:
                c_e += 10
            else:
                c_e = 697
        elif c_wise == 703:
            if c_e <= 693:
                c_e += 10
            else:
                c_e = 703
        else:
            if c_e <= 709:
                c_e += 10
            else:
                c_e = 719


    a_flag = 0
    b_flag = 0
    c_flag = 0

    if (a_e-a_s)<=64:
        if (a_s + a_e) // 2 < 32:
            a_s = 0
            a_e = 64
        elif (a_s + a_e) // 2 > a_wise - 32:
            a_s = a_wise - 64
            a_e = a_wise
        else:
            center = (a_s + a_e) // 2
            a_s = center - 32
            a_e = center + 32
        a_flag=1
    else:
        a_flag=2
        a_s_1 = a_s
        a_e_1 = a_s+64
        a_s_2 = a_e-64
        a_e_2 = a_e

    if (b_e-b_s)<=256:
        if (b_s+b_e)//2 <128:
            b_s = 0
            b_e = 256
        elif (b_s+b_e)//2>b_wise-128:
            b_s = b_wise -256
            b_e = b_wise
        else:
            center = (b_s+b_e)//2
            b_s = center-128
            b_e = center+128
        b_flag=1
    else:
        b_flag=2
        b_s_1 = b_s
        b_e_1 = b_s+256
        b_s_2 = b_e-256
        b_e_2 = b_e

    if (c_e-c_s)<=256:
        if (c_s+c_e)//2 <128:
            c_s = 0
            c_e = 256
        elif (c_s+c_e)//2>c_wise-128:
            c_s = c_wise -256
            c_e = c_wise
        else:
            center = (c_s+c_e)//2
            c_s = center-128
            c_e = center+128
        c_flag=1
    else:
        c_flag=2
        c_s_1 = c_s
        c_e_1 = c_s+256
        c_s_2 = c_e-256
        c_e_2 = c_e

    count = 0
    if a_flag==1:
        if b_flag==1:
            if c_flag ==1:
                _dir = dir + '/' + name + '_0'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s:a_e,b_s:b_e,c_s:c_e])
                np.save(_dir + '/label.npy', label_np[a_s:a_e,b_s:b_e,c_s:c_e])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir+'/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                    format(a_wise,b_wise,c_wise,a_s,b_s,c_s))

            else:
                _dir = dir + '/' + name + '_0'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s:a_e, b_s:b_e, c_s_1:c_e_1])
                np.save(_dir + '/label.npy', label_np[a_s:a_e, b_s:b_e, c_s_1:c_e_1])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s, b_s, c_s_1))
                _dir = dir + '/' + name + '_1'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s:a_e, b_s:b_e, c_s_2:c_e_2])
                np.save(_dir + '/label.npy', label_np[a_s:a_e, b_s:b_e, c_s_2:c_e_2])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s, b_s, c_s_2))
        else:
            if c_flag == 1:
                _dir = dir + '/' + name + '_0'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s:a_e, b_s_1:b_e_1, c_s:c_e])
                np.save(_dir + '/label.npy', label_np[a_s:a_e, b_s_1:b_e_1, c_s:c_e])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s, b_s_1, c_s))
                _dir = dir + '/' + name + '_1'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s:a_e, b_s_2:b_e_2, c_s:c_e])
                np.save(_dir + '/label.npy', label_np[a_s:a_e, b_s_2:b_e_2, c_s:c_e])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s, b_s_2, c_s))
            else:
                _dir = dir + '/' + name + '_0'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s:a_e, b_s_1:b_e_1, c_s_1:c_e_1])
                np.save(_dir + '/label.npy', label_np[a_s:a_e, b_s_1:b_e_1, c_s_1:c_e_1])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s, b_s_1, c_s_1))
                _dir = dir + '/' + name + '_1'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s:a_e, b_s_2:b_e_2, c_s_1:c_e_1])
                np.save(_dir + '/label.npy', label_np[a_s:a_e, b_s_2:b_e_2, c_s_1:c_e_1])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s, b_s_2, c_s_1))
                _dir = dir + '/' + name + '_2'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s:a_e, b_s_1:b_e_1, c_s_2:c_e_2])
                np.save(_dir + '/label.npy', label_np[a_s:a_e, b_s_1:b_e_1, c_s_2:c_e_2])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s, b_s_1, c_s_2))
                _dir = dir + '/' + name + '_3'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s:a_e, b_s_2:b_e_2, c_s_2:c_e_2])
                np.save(_dir + '/label.npy', label_np[a_s:a_e, b_s_2:b_e_2, c_s_2:c_e_2])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s, b_s_2, c_s_2))
    else:
        if b_flag==1:
            if c_flag==1:
                _dir = dir + '/' + name + '_0'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s_1:a_e_1, b_s:b_e, c_s:c_e])
                np.save(_dir + '/label.npy', label_np[a_s_1:a_e_1, b_s:b_e, c_s:c_e])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s_1, b_s, c_s))
                _dir = dir + '/' + name + '_1'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s_2:a_e_2, b_s:b_e, c_s:c_e])
                np.save(_dir + '/label.npy', label_np[a_s_2:a_e_2, b_s:b_e, c_s:c_e])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s_2, b_s, c_s))
            else:
                _dir = dir + '/' + name + '_0'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s_1:a_e_1, b_s:b_e, c_s_1:c_e_1])
                np.save(_dir + '/label.npy', label_np[a_s_1:a_e_1, b_s:b_e, c_s_1:c_e_1])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s_1, b_s, c_s_1))
                _dir = dir + '/' + name + '_1'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s_1:a_e_1, b_s:b_e, c_s_2:c_e_2])
                np.save(_dir + '/label.npy', label_np[a_s_1:a_e_1, b_s:b_e, c_s_2:c_e_2])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s_1, b_s, c_s_2))
                _dir = dir + '/' + name + '_2'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s_2:a_e_2, b_s:b_e, c_s_1:c_e_1])
                np.save(_dir + '/label.npy', label_np[a_s_2:a_e_2, b_s:b_e, c_s_1:c_e_1])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s_2, b_s, c_s_1))
                _dir = dir + '/' + name + '_3'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s_2:a_e_2, b_s:b_e, c_s_2:c_e_2])
                np.save(_dir + '/label.npy', label_np[a_s_2:a_e_2, b_s:b_e, c_s_2:c_e_2])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s_2, b_s, c_s_2))
        else:
            if c_flag==1:
                _dir = dir + '/' + name + '_0'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s_1:a_e_1, b_s_1:b_e_1, c_s:c_e])
                np.save(_dir + '/label.npy', label_np[a_s_1:a_e_1, b_s_1:b_e_1, c_s:c_e])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s_1, b_s_1, c_s))
                _dir = dir + '/' + name + '_1'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s_1:a_e_1, b_s_2:b_e_2, c_s:c_e])
                np.save(_dir + '/label.npy', label_np[a_s_1:a_e_1, b_s_2:b_e_2, c_s:c_e])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s_1, b_s_2, c_s))
                _dir = dir + '/' + name + '_2'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s_2:a_e_2, b_s_1:b_e_1, c_s:c_e])
                np.save(_dir + '/label.npy', label_np[a_s_2:a_e_2, b_s_1:b_e_1, c_s:c_e])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s_2, b_s_1, c_s))
                _dir = dir + '/' + name + '_3'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s_2:a_e_2, b_s_2:b_e_2, c_s:c_e])
                np.save(_dir + '/label.npy', label_np[a_s_2:a_e_2, b_s_2:b_e_2, c_s:c_e])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s_2, b_s_2, c_s))
            else:
                _dir = dir + '/' + name + '_0'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s_1:a_e_1, b_s_1:b_e_1, c_s_1:c_e_1])
                np.save(_dir + '/label.npy', label_np[a_s_1:a_e_1, b_s_1:b_e_1, c_s_1:c_e_1])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s_1, b_s_1, c_s_1))
                _dir = dir + '/' + name + '_1'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s_1:a_e_1, b_s_1:b_e_1, c_s_2:c_e_2])
                np.save(_dir + '/label.npy', label_np[a_s_1:a_e_1, b_s_1:b_e_1, c_s_2:c_e_2])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s_1, b_s_1, c_s_2))
                _dir = dir + '/' + name + '_2'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s_1:a_e_1, b_s_2:b_e_2, c_s_1:c_e_1])
                np.save(_dir + '/label.npy', label_np[a_s_1:a_e_1, b_s_2:b_e_2, c_s_1:c_e_1])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s_1, b_s_2, c_s_1))
                _dir = dir + '/' + name + '_3'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s_1:a_e_1, b_s_2:b_e_2, c_s_2:c_e_2])
                np.save(_dir + '/label.npy', label_np[a_s_1:a_e_1, b_s_2:b_e_2, c_s_2:c_e_2])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s_1, b_s_2, c_s_2))
                _dir = dir + '/' + name + '_4'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s_2:a_e_2, b_s_1:b_e_1, c_s_1:c_e_1])
                np.save(_dir + '/label.npy', label_np[a_s_2:a_e_2, b_s_1:b_e_1, c_s_1:c_e_1])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s_2, b_s_1, c_s_1))
                _dir = dir + '/' + name + '_5'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s_2:a_e_2, b_s_1:b_e_1, c_s_2:c_e_2])
                np.save(_dir + '/label.npy', label_np[a_s_2:a_e_2, b_s_1:b_e_1, c_s_2:c_e_2])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s_2, b_s_1, c_s_2))
                _dir = dir + '/' + name + '_6'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s_2:a_e_2, b_s_2:b_e_2, c_s_1:c_e_1])
                np.save(_dir + '/label.npy', label_np[a_s_2:a_e_2, b_s_2:b_e_2, c_s_1:c_e_1])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s_2, b_s_2, c_s_1))
                _dir = dir + '/' + name + '_7'
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)
                np.save(_dir + '/image.npy', img_np[a_s_2:a_e_2, b_s_2:b_e_2, c_s_2:c_e_2])
                np.save(_dir + '/label.npy', label_np[a_s_2:a_e_2, b_s_2:b_e_2, c_s_2:c_e_2])
                target = open(_dir + '/label.txt', 'w')  # 打开目的文件
                target.write(indate)
                size = _dir + '/size.txt'
                open(size, 'w').write(
                    "{} {} {} {} {} {}\n".
                        format(a_wise, b_wise, c_wise, a_s_2, b_s_2, c_s_2))


def Flip(name,dir,img_np,label_np,indate):
    print(name)
    dir = dir+'/flip1/'+name
    if not os.path.isdir(dir):
        os.makedirs(dir)
    print(img_np.shape)
    img_np = img_np.transpose(0,2,1)
    print(img_np.shape)
    label_np = label_np.transpose(0,2,1)
    np.save(dir+'/imgae.npy',img_np)
    np.save(dir+'/label.npy',label_np)
    target = open(dir+'/label.txt', 'w')  # 打开目的文件
    target.write(indate)
def Flip1(name,dir,img_np,label_np,indate):
    print(name)
    dir = dir+'/flip2/'+name
    if not os.path.isdir(dir):
        os.makedirs(dir)
    img_np_f = np.empty_like(img_np)
    label_np_f = np.empty_like(label_np)
    for i in range(img_np.shape[0]):
        img_np_f[img_np.shape[0]-i-1,:,:]= img_np[i,:,:]
        label_np_f[img_np.shape[0]-i-1,:,:] = label_np[i,:,:]
    np.save(dir+'/imgae.npy',img_np_f)
    np.save(dir+'/label.npy',label_np_f)
    target = open(dir+'/label.txt', 'w')  # 打开目的文件
    target.write(indate)
def Gauss(name,dir,img_np,label_np,indate):
    dir = dir+'/gauss/'+name
    if not os.path.isdir(dir):
        os.makedirs(dir)
    SNR = 5
    noise = np.random.randn(img_np.shape[0], img_np.shape[1],img_np.shape[2])  # 产生N(0,1)噪声数据
    noise = noise - np.mean(noise)  # 均值为0
    signal_power = np.linalg.norm(img_np) ** 2 / img_np.size  # 此处是信号的std**2
    noise_variance = signal_power / np.power(10, (SNR / 10))  # 此处是噪声的std**2
    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise  ##此处是噪声的std**2
    img_np_noise = noise + img_np
    np.save(dir + '/imgae.npy', img_np_noise)
    np.save(dir + '/label.npy', label_np)
    target = open(dir + '/label.txt', 'w')  # 打开目的文件
    target.write(indate)
def Patches(name,dir,img_np,label_np,indate):
    dir = dir+'/patch/'+name

    a, b, c = [], [], []
    for i in range(label_np.shape[0]):
        (mean, stddv) = cv2.meanStdDev(label_np[i, :, :])
        if mean > 0:
            a.append(i)
    for i in range(label_np.shape[1]):
        (mean, stddv) = cv2.meanStdDev(label_np[:, i, :])
        if mean > 0:
            b.append(i)
    for i in range(label_np.shape[2]):
        (mean, stddv) = cv2.meanStdDev(label_np[:, :, i])
        if mean > 0:
            c.append(i)
    a_s = a[0]
    a_e = a[len(a) - 1]
    b_s = b[0]
    b_e = b[len(b) - 1]
    c_s = c[0]
    c_e = c[len(c) - 1]

    patch_z = [0,64,128,192,256,318] #-2
    patch_x_1 = [0,256,443] #-69
    patch_x_2 = [0,256,512]
    patch_x_3 = [0,256,507] #-5
    patch_x_4 = [0,256,489] #-23

    patch_y_1 = [0,256,477,697]#-35,-36
    patch_y_2 = [0,256,480,703]#-32,-33
    patch_y_3 = [0,256,488,719]#-24,-25
    patch_y_4 = [0,256,474,692]  # -38,-38
    img_patch = np.empty((30, 64, 256, 256))
    label_patch = np.empty((30, 64, 256, 256))
    label = np.empty(30,dtype=int)
    count = 0
    for i in range(5):
        for j in range(2):
            for k in range(3):
                if i == 4:
                    p_z_s = patch_z[i] - 2
                    p_z_e = patch_z[i + 1]
                else:
                    p_z_s = patch_z[i]
                    p_z_e = patch_z[i + 1]
                if j == 1:
                    if img_np.shape[1]==443:

                        p_x_s = patch_x_1[j] - 69
                        p_x_e = patch_x_1[j + 1]
                    elif img_np.shape[1]==595:

                        p_x_s = patch_x_2[j]
                        p_x_e = patch_x_2[j + 1]
                    elif img_np.shape[1]==507:

                        p_x_s = patch_x_3[j]-5
                        p_x_e = patch_x_3[j + 1]
                    else:

                        p_x_s = patch_x_4[j] -23
                        p_x_e = patch_x_4[j + 1]
                else:
                    if img_np.shape[1] == 443:
                        p_x_s = patch_x_1[j]
                        p_x_e = patch_x_1[j + 1]
                    elif img_np.shape[1] == 595:
                        p_x_s = patch_x_2[j]
                        p_x_e = patch_x_2[j + 1]
                    elif img_np.shape[1] == 507:
                        p_x_s = patch_x_3[j]
                        p_x_e = patch_x_3[j + 1]
                    else:
                        p_x_s = patch_x_4[j]
                        p_x_e = patch_x_4[j + 1]
                if k == 1:
                    if img_np.shape[1] == 443:
                        p_y_s = patch_y_1[k]-35
                        p_y_e = patch_y_1[k + 1]
                    elif img_np.shape[1] == 595:
                        p_y_s = patch_y_2[k]-32
                        p_y_e = patch_y_2[k + 1]
                    elif img_np.shape[1] == 507:
                        p_y_s = patch_y_3[k]-24
                        p_y_e = patch_y_3[k + 1]
                    else:
                        p_y_s = patch_y_4[k]-38
                        p_y_e = patch_y_4[k + 1]
                elif k == 2:
                    if img_np.shape[1] == 443:
                        p_y_s = patch_y_1[k] - 36
                        p_y_e = patch_y_1[k + 1]
                    elif img_np.shape[1] == 595:
                        p_y_s = patch_y_2[k] - 33
                        p_y_e = patch_y_2[k + 1]
                    elif img_np.shape[1] == 507:
                        p_y_s = patch_y_3[k] - 25
                        p_y_e = patch_y_3[k + 1]
                    else:
                        p_y_s = patch_y_4[k] - 38
                        p_y_e = patch_y_4[k + 1]
                else:
                    if img_np.shape[1] == 443:
                        p_y_s = patch_y_1[k]
                        p_y_e = patch_y_1[k + 1]
                    elif img_np.shape[1] == 595:
                        p_y_s = patch_y_2[k]
                        p_y_e = patch_y_2[k + 1]
                    elif img_np.shape[1] == 507:
                        p_y_s = patch_y_3[k]
                        p_y_e = patch_y_3[k + 1]
                    else:
                        p_y_s = patch_y_4[k]
                        p_y_e = patch_y_4[k + 1]
                img_patch[count] = img_np[p_z_s:p_z_e, p_x_s:p_x_e, p_y_s:p_y_e]
                if ((p_z_e <= a_e and p_z_e >= a_s) or (p_z_s <= a_e and p_z_s >= a_s) or (
                        p_z_s <= a_s and p_z_e >= a_e)) and (
                        (p_x_e <= b_e and p_x_e >= b_s) or (p_x_s <= b_e and p_x_s >= b_s) or (
                        p_x_s <= b_s and p_x_e >= b_e)) and (
                        (p_y_e <= c_e and p_y_e >= c_s) or (p_y_s <= c_e and p_y_s >= c_s) or (
                        p_y_s <= c_s and p_y_e >= c_e)):
                    label[count] = indate
                else:
                    label[count] = 0
                count += 1
    if not os.path.isdir(dir):
        os.makedirs(dir)
    target = open(dir + '/number.txt', 'w')  # 打开目的文件
    target.write(str(img_np.shape[1]))
    for m in range(30):
        _dir = dir + '/' + str(m)
        if not os.path.isdir(_dir):
            os.makedirs(_dir)
        np.save(_dir + '/imgae.npy', img_patch[m])
        np.save(_dir + '/label.npy', label_patch[m])
        target = open(_dir + '/label.txt', 'w')  # 打开目的文件
        target.write(str(label[m]))

dir = "/home/ubuntu/liuyiyao/3D_breast_Seg/Dataset/best_data1"
img_root = '/home/ubuntu/liuyiyao/3D_breast_Seg/Dataset/breast_input/img'
mask_root = '/home/ubuntu/liuyiyao/3D_breast_Seg/Dataset/breast_input/label'
label_root="/home/ubuntu/liuyiyao/3D_breast_Seg/Dataset/vet_dataset"
mask_names = os.listdir(mask_root)
j=0
for name in mask_names:

    mask_data = sitk.ReadImage(mask_root + '/' + name)
    mask_np = sitk.GetArrayFromImage(mask_data)
    img_data = sitk.ReadImage(img_root + '/' + name)
    img_np = sitk.GetArrayFromImage(img_data)
    print(name.split('.')[0])
    source = open(label_root + '/' + name.split('.')[0] + '/label.txt')  # 打开源文件
    indate = source.read()  # 显示所有源文件内容
    Slice(name.split('.')[0],dir,img_np,mask_np,indate)
    j+=1

