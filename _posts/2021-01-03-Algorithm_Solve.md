---
layout: post
title: "[프로그래머스] 그리디 - 큰 수 만들기 (Python)"
date: 2021-01-03 08:04:23 +0900
category: Algorithm
---

# 문제 설명
어떤 숫자에서 k개의 수를 제거했을 때 얻을 수 있는 가장 큰 숫자를 구하려 합니다.  
예를 들어, 숫자 1924에서 수 두 개를 제거하면 [19, 12, 14, 92, 94, 24] 를 만들 수 있습니다. 이 중 가장 큰 숫자는 94 입니다.  
문자열 형식으로 숫자 number와 제거할 수의 개수 k가 solution 함수의 매개변수로 주어집니다. number에서 k 개의 수를 제거했을 때 만들 수 있는 수 중 가장 큰 숫자를 문자열 형태로 return 하도록 solution 함수를 완성하세요.  

# 제한 조건
- number는 1자리 이상, 1,000,000자리 이하인 숫자입니다.
- k는 1 이상 number의 자릿수 미만인 자연수입니다.
<br>
<br>
### 입출력 예
|number|k|return|
|---|---|---|
|"1924"|2|"94"|
|"1231234"|3|"3234"|
|"4177252841"|4|"775841"|

<br>
<br>

1. 모든 케이스 확인
<br>
{% highlight python %}
def solution(number, k):
    while k:
        group = [int(number[:i]+number[i+1:]) for i in range(len(number))]
        number = str(max(group))
        k-=1
    return number
{% endhighlight %}
<br>
![image](/public/img/큰수만들기1.png){: width="100%" height="100%"}{: .center}
<br>
모든 케이스를 다 검사해서 시간초과가 났다.  
1000000(number) * 1000000(k)라 10억을 가뿐히 넘어서 그런듯..  
<br>
