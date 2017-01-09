知乎爬虫————实现验证码自动识别

1.
使用keras框架搭建小型VGG卷积神经网络：
网络分为四个卷积层（卷积核3X3），两个采样层。一个全连接层
2.
从www.zhihu.com/captcha.gif爬取验证码图片作为训练样本
3.
分别训练出一个切割器和一个识别器
4.
网络模型和网络权重保存在.h5文件中可直接读取
5.
只需修改zhihu.py文件中的登录邮箱及密码，运行zhihu.py即可 
当然前提是要先安装好keras：)
6.
keras的安装可参考https://keras-cn.readthedocs.io/en/latest/
