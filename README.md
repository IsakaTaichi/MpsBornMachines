# MpsBornMachines

行列積状態による生成モデル

論文  
Unsupervised Generative Modeling Using Matrix Product States   
https://arxiv.org/pdf/1709.01662.pdf  
github  
https://github.com/congzlwag/UnsupGenModbyMPS  

論文をベースにpytorchによる実装  


データ$x$の確率分布  
$$
p(x)=\frac{|\psi(x)|^2}{z}
$$

$\psi(x)$は、$A_{v_i}^{(i)}$のMPS状態  
$$
\psi(x) = Tr(A_{v_1}^{(1)}A_{v_2}^{(2)}A_{v_3}^{(3)} ...A_{v_n}^{(n)})
$$

データセットの負の対数尤度LLMを誤差関数として、パラメータ更新する  

$$
L = - \sum_i \frac{1}{|T|} ln p(x_i)
$$
