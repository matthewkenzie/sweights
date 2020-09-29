## A simple class for Dalitz analyses
import numpy as np

class dalitz:

    def __init__(self, md, ma, mb, mc):
        self.fMa = ma
        self.fMb = mb
        self.fMc = mc
        self.fMd = md
        self.fM2a = ma*ma
        self.fM2b = mb*mb
        self.fM2c = mc*mc
        self.fM2d = md*md
        self.fM2sum = self.fM2a + self.fM2b + self.fM2c + self.fM2d
        self.abrange = (np.floor((self.fMa + self.fMb)**2) , np.ceil((self.fMd - self.fMc)**2))
        self.bcrange = (np.floor((self.fMb + self.fMc)**2) , np.ceil((self.fMd - self.fMa)**2))
        self.acrange = (np.floor((self.fMa + self.fMc)**2) , np.ceil((self.fMd - self.fMb)**2))

    def in_kine_limits(self, m2ab, m2ac):
        #passes = (self.fM2sum < m2ab+m2ac)
        #passes = (m2ab > self.abrange[0]) & (m2ab < self.abrange[1])
        #passes = passes & (m2ab > self.abrange[0]) & (m2ab < self.abrange[1])
        m2bc = self.fM2sum - m2ab -m2ac
        #m2bc[ m2bc < 0 ] = 0
        mab = m2ab**0.5
        mac = m2ac**0.5
        mbc = m2bc**0.5

        p2a = 0.25/self.fM2d*(self.fM2d-(mbc+self.fMa)**2)*(self.fM2d-(mbc-self.fMa)**2)
        p2b = 0.25/self.fM2d*(self.fM2d-(mac+self.fMb)**2)*(self.fM2d-(mac-self.fMb)**2)
        p2c = 0.25/self.fM2d*(self.fM2d-(mab+self.fMc)**2)*(self.fM2d-(mab-self.fMc)**2)
        #print( p2a, p2b, p2c)
        #passes = passes & (p2a > 0) & (p2b > 0) & (p2c > 0)

        eb = (m2ab-self.fM2a+self.fM2b)/2./mab
        ec = (self.fM2d-m2ab-self.fM2c)/2./mab
        if ( eb < self.fMb ) or ( ec < self.fMc ): return False
        #passes = passes & (eb > self.fMb) & (ec > self.fMc)

        pb = np.sqrt(eb**2-self.fM2b)
        pc = np.sqrt(ec**2-self.fM2c)
        e2sum = (eb+ec)**2
        m2bc_max = e2sum-(pb-pc)**2
        m2bc_min = e2sum-(pb+pc)**2
        if m2bc < m2bc_min or m2bc > m2bc_max: return False
        if (p2a>0 and p2b>0 and p2c>0): return True
        #passes = passes & (m2bc > m2bc_min) & (m2bc < m2bc_max)
        return False

    def dp_contour(self, x, y, orientation=1323):

        mParent = self.fMd
        if orientation == 1323:
            mi = self.fMa
            mj = self.fMb
            mk = self.fMc
        elif orientation == 2313:
            mi = self.fMb
            mj = self.fMa
            mk = self.fMc
        elif orientation == 1213:
            mi = self.fMb
            mj = self.fMc
            mk = self.fMa
        elif orientation == 1312:
            mi = self.fMc
            mj = self.fMb
            mk = self.fMa
        elif orientation == 1223:
            mi = self.fMa
            mj = self.fMc
            mk = self.fMb
        elif orientation == 2312:
            mi = self.fMc
            mj = self.fMa
            mk = self.fMb
        else:
            raise RuntimeError('Invalid orientation', orientation)

        mik = np.sqrt(x)
        mjk = np.sqrt(y)

        ejcmsik = (mParent*mParent-mj*mj-mik*mik)/(2.0*mik)
        ekcmsik = (mik*mik+mk*mk-mi*mi)/(2.0*mik)

        pj = np.sqrt(np.abs(ejcmsik*ejcmsik-mj*mj))
        pk = np.sqrt(np.abs(ekcmsik*ekcmsik-mk*mk))

        coshelik = (mjk*mjk - mk*mk - mj*mj - 2.0*ejcmsik*ekcmsik)/(2.0*pj*pk)
        coshelikSq = coshelik*coshelik

        coshelikSq = np.where( ekcmsik < mk , 2. , coshelikSq )
        coshelikSq = np.where( ejcmsik < mj  , 2. , coshelikSq )
        return np.where( np.isnan(coshelikSq), 2., coshelikSq )

    def psgen(self, size=1):

        res = np.empty((0,2))
        for i in range(size):
            x = np.random.uniform( *self.abrange )
            y = np.random.uniform( *self.acrange )
            while not self.in_kine_limits(x,y):
                x = np.random.uniform( *self.abrange )
                y = np.random.uniform( *self.acrange )

            res = np.append(res,[[x,y]],axis=0)
        return res
