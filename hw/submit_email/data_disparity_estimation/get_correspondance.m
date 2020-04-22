function [x_p, y_p] = get_correspondance(h, x, y)
    x_p = (h(1)*x + h(2)*y + h(3))/(h(7)*x + h(8)*y + 1);
    y_p = (h(4)*x + h(5)*y + h(6))/(h(7)*x + h(8)*y + 1);
end