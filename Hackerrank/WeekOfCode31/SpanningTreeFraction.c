// Author : Amit Kumar Jaiswal

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>
#include <stdbool.h>

struct line
{
    struct point * dest;
    int a, b;
    struct line * nxt;
};

struct point
{
    int index;
    struct point * belong;
    struct point * pre;
    struct point * nxt;
    struct line * lines;
};

int max_a, max_b;
double max = 0;

void addToSet(struct point * set, struct point * p)
{
    if (set->nxt)
    {
        set->nxt->pre = p;
    }
    p->nxt = set->nxt;
    p->pre = set;
    set->nxt = p;
    p->belong = set;
}

void delFromSet(struct point * set, struct point * p)
{
    if (p->nxt)
    {
        p->nxt->pre = p->pre;
    }
    p->pre->nxt = p->nxt;
}

void rfunc(struct point * p, struct line * start_line, struct point * unuse, struct point * used, int a, int b)
{
//    printf("p->index = %d\n", p->index);
    if (!unuse->nxt)
    {
//        printf("%d/%d\n", a, b);
        if ((double)a/(double)b > max)
        {
            max = (double)a/(double)b;
            max_a = a;
            max_b = b;
        }
    }
    else {
        struct line * l = start_line;
        if (l) {
            do {
                if (unuse == l->dest->belong) {
                    struct point * p_d = l->dest;
                    delFromSet(unuse, p_d);
                    addToSet(used, p_d);

                    rfunc(p_d, p_d->lines, unuse, used, a + l->a, b + l->b);
                    rfunc(p, l->nxt, unuse, used, a + l->a, b + l->b);

                    delFromSet(used, p_d);
                    addToSet(unuse, p_d);

                    rfunc(p, l->nxt, unuse, used, a, b);
                }
            } while(0 != (l = l->nxt));
        }
    }
}

int main(){
    int n;
    int m;
    scanf("%d %d",&n,&m);

    struct point * * points = malloc(sizeof(struct point *) * n);
    memset(points, 0, sizeof(struct point *) * n);

    struct point * unuse = malloc(sizeof(struct point));
    struct point * used = malloc(sizeof(struct point));
    memset(unuse, 0, sizeof(struct point));
    memset(used, 0, sizeof(struct point));

    for(int a0 = 0; a0 < m; a0++){
        int u;
        int v;
        int a;
        int b;
        scanf("%d %d %d %d",&u,&v,&a,&b);
        // Write Your Code Here
        if (u == v) {
            // loop
        }
        else {
            struct point * p_u = points[u];
            if (!p_u) {
                p_u = malloc(sizeof(struct point));
                memset(p_u, 0, sizeof(struct point));
                p_u->index = u;
                points[u] = p_u;
                addToSet(unuse, p_u);
            }

            struct point * p_v = points[v];
            if (!p_v) {
                p_v = malloc(sizeof(struct point));
                memset(p_v, 0, sizeof(struct point));
                p_v->index = v;
                points[v] = p_v;
                addToSet(unuse, p_v);
            }

            {
                struct line * l = malloc(sizeof(struct line));
                l->a = a;
                l->b = b;
                l->dest = p_v;
                l->nxt = p_u->lines;
                p_u->lines = l;
            }

            {
                struct line * l = malloc(sizeof(struct line));
                l->a = a;
                l->b = b;
                l->dest = p_u;
                l->nxt = p_v->lines;
                p_v->lines = l;
            }
        }
    }

    struct point * first = unuse->nxt;
    delFromSet(unuse, first);
    addToSet(used, first);

    rfunc(first, first->lines, unuse, used, 0, 0);

    int step = 2;
    while (step <= max_a && step <= max_b) {
        if (0 == (max_a % step) && (0 == (max_b % step))) {
            max_a /= step;
            max_b /= step;
            step = 2;
        }
        else {
            step ++;
        }
    }

    printf("%d/%d\n", max_a, max_b);

    free(points);

    return 0;
}
