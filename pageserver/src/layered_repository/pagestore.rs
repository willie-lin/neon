use bumpalo::Bump;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::sync::atomic::AtomicPtr;
use std::sync::atomic::Ordering;

use crate::layered_repository::inmemory_layer::InMemoryLayerInner;

pub struct BumpLock
{
    lock: RwLock<()>,
    
    bump: AtomicPtr<Bump>,

    inner: AtomicPtr<InMemoryLayerInner<'static>>,
}

pub struct BumpLockReadGuard<'a>
{
    _lock_guard: RwLockReadGuard<'a, ()>,
    inner: &'a InMemoryLayerInner<'a>,
}

impl<'a> std::ops::Deref for BumpLockReadGuard<'a> {
    type Target = InMemoryLayerInner<'a>;

    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

pub struct BumpLockWriteGuard<'a> {
    _lock_guard: RwLockWriteGuard<'a, ()>,
    inner: &'a mut InMemoryLayerInner<'a>,
    bump: &'a Bump,
}

impl<'a> std::ops::Deref for BumpLockWriteGuard<'a> {
    type Target = InMemoryLayerInner<'a>;

    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

impl<'a> std::ops::DerefMut for BumpLockWriteGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner
    }
}

impl<'a> BumpLockWriteGuard<'a> {
    pub fn bump(&self) -> &'a Bump {
        self.bump
    }
}

impl<'a> BumpLock
{
    pub fn read(&'a self) -> BumpLockReadGuard<'a> {

        let lock_guard = self.lock.read().unwrap();

        let inner = unsafe { self.inner.load(Ordering::Relaxed).as_ref() }.unwrap();

        let guard = BumpLockReadGuard {
            _lock_guard: lock_guard,
            inner,
        };

        guard
    }

    pub fn write(&'a self) -> BumpLockWriteGuard<'a>
    {
        let lock_guard = self.lock.write().unwrap();

        let inner = unsafe { self.inner.load(Ordering::Relaxed).as_mut() }.unwrap();

        let inner =
        unsafe {
            std::mem::transmute::<&mut InMemoryLayerInner<'static>, &mut InMemoryLayerInner<'a>>(inner)
        };

        let bump = unsafe {self.bump.load(Ordering::Relaxed).as_ref() }.unwrap();

        let guard = BumpLockWriteGuard {
            _lock_guard: lock_guard,
            inner,
            bump,
        };

        guard
    }

    pub fn new() -> Self {
        let bump = Box::new(Bump::new());

        let inner = Box::new(InMemoryLayerInner::new());

        BumpLock {
            lock: RwLock::new(()),
            bump: AtomicPtr::new(Box::into_raw(bump)),
            inner: AtomicPtr::new(Box::into_raw(inner)),
        }
    }
}
