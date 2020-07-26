# PST
Part-based Semantic Transform for Few-shot Semantic Segmentation

## Overview
- `networks/` contains the implementation of the PST(`PST_net.py`);
- `models/` contains the backbone, semantic decomposition module;
- `utils/` contains the semantic match module;

## Dependencies
python == 3.7,
pytorch1.0,

torchvision,
pillow,
opencv-python,
pandas,
matplotlib,
scikit-image

## Performance
Performance of k shot semantic segmentation on Pascal-5<sup>i</sup>
<table>
    <tr>
        <td>Setting</td>
        <td>Backbone</td>
        <td>Method</td>
        <td>Pascal-5<sup>0</sup></td>
        <td>Pascal-5<sup>1</sup></td>
        <td>Pascal-5<sup>2</sup></td>
        <td>Pascal-5<sup>3</sup></td>
        <td>Mean</td>
    </tr>
    <tr>
        <td rowspan="3">1-shot</td>
        <td>VGG16</td>
        <td>PST</td>
        <td>48.89</td>
        <td>64.65</td>
        <td>51.44</td>
        <td>47.52</td>
        <td>53.12</td>
    </tr>
    <tr>
        <td rowspan="2">Resnet50</td>
        <td>PMMs</td>
        <td>51.98</td>
        <td>67.54</td>
        <td>51.54</td>
        <td>49.81</td>
        <td>55.22</td>
    </tr>
    <tr>
        <td>PST</td>
        <td>52.66</td>
        <td>67.13</td>
        <td>53.23</td>
        <td>51.48</td>
        <td>56.14</td>
    </tr>
    <tr>
        <td rowspan="3">5-shot</td>
        <td>VGG16</td>
        <td>PST</td>
        <td>51.14</td>
        <td>65.27</td>
        <td>52.83</td>
        <td>48.51</td>
        <td>54.44</td>
    </tr>
    <tr>
        <td rowspan="2">Resnet50</td>
        <td>PMMs</td>
        <td>55.03</td>
        <td>68.22</td>
        <td>52.89</td>
        <td>51.11</td>
        <td>56.81</td>
    </tr>
    <tr>
        <td>PST</td>
        <td>54.93</td>
        <td>68.69</td>
        <td>53.77</td>
        <td>51.76</td>
        <td>57.29</td>
    </tr>
</table>


Performance of k shot semantic segmentation on MS COCO
<table>
    <tr>
        <td>Setting</td>
        <td>Method</td>
        <td>COCO-20<sup>0</sup></td>
        <td>COCO-20<sup>1</sup></td>
        <td>COCO-20<sup>2</sup></td>
        <td>COCO-20<sup>3</sup></td>
        <td>Mean</td>
    </tr>
    <tr>
        <td rowspan="2">1-shot</td>
        <td>PMMs</td>
        <td>29.28</td>
        <td>34.81</td>
        <td>27.08</td>
        <td>27.27</td>
        <td>29.61</td>
    </tr>
    <tr>
        <td>PST</td>
        <td>30.37</td>
        <td>37.51</td>
        <td>30.22</td>
        <td>32.56</td>
        <td>32.67</td>
    </tr>
    <tr>
        <td rowspan="2">5-shot</td>
        <td>PMMs</td>
        <td>33.00</td>
        <td>40.55</td>
        <td>30.29</td>
        <td>33.27</td>
        <td>34.28</td>
    </tr>
    <tr>
        <td>PST</td>
        <td>34.06</td>
        <td>42.20</td>
        <td>34.74</td>
        <td>38.95</td>
        <td>37.49</td>
    </tr>
</table>
